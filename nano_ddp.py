from __future__ import annotations

import functools
from collections.abc import Callable

import torch
import torch.distributed as dist
from torch.nn import Module


class _NanoDDPBase(Module):
    """Shared: module handle, process group, managed params, init broadcast, forward."""

    def __init__(
        self,
        module: Module,
        *,
        device: torch.device,
        process_group: dist.ProcessGroup | None = None,
    ) -> None:
        super().__init__()
        self.module = module
        self.device = device
        self.process_group = (
            process_group if process_group is not None else dist.group.WORLD
        )

        self._world_size: int = dist.get_world_size(self.process_group)
        self._rank: int = dist.get_rank(self.process_group)

        # param_name and parameter
        self._managed_params: list[tuple[str, torch.nn.Parameter]] = []

        self._build_managed_parameters()
        self._broadcast_initial_parameters()

    def _build_managed_parameters(self) -> None:
        """Traverse named_parameters (dedupe shared weights, requires_grad only);"""
        memo: set[torch.nn.Parameter] = set()
        out: list[tuple[str, torch.nn.Parameter]] = []
        for param_name, param in self.module.named_parameters():
            if not param.requires_grad:
                continue
            if param not in memo and not memo.add(param):
                out.append((param_name, param))
        self._managed_params = out
        if not self._managed_params:
            raise RuntimeError(
                "NanoDDP expects at least one parameter with requires_grad=True."
            )

    def _broadcast_initial_parameters(self) -> None:
        """Broadcast initial parameters from rank 0."""
        for _param_name, param in self._managed_params:
            dist.broadcast(param, src=0, group=self.process_group)

    def forward(self, *args: object, **kwargs: object) -> object:
        return self.module(*args, **kwargs)

    def _register_per_param_post_accumulate_hooks(
        self,
        handles: list,
        on_grad_ready: Callable[[int, torch.Tensor], torch.Tensor],
    ) -> None:
        """Register ``on_grad_ready(param_index, grad)`` on each managed parameter."""
        for idx, (_param_name, param) in enumerate(self._managed_params):
            handles.append(
                param.register_post_accumulate_grad_hook(
                    functools.partial(on_grad_ready, idx)
                )
            )

    def _register_finalize_when_all_grads_ready(
        self, handles: list, finalize: Callable[[], None]
    ) -> None:
        """Append a hook that calls ``finalize()`` after all in-graph param grads are ready."""
        params = [param for _name, param in self._managed_params]

        def _on_all_grads_accumulated(
            _grads: tuple[torch.Tensor | None, ...],
        ) -> None:
            del _grads
            finalize()

        handles.append(
            torch.autograd.graph.register_multi_grad_hook(
                params, _on_all_grads_accumulated
            )
        )


class NanoDDPV1(_NanoDDPBase):
    """V1: all reduce all params after whole backward is finished
    """

    def __init__(
        self,
        module: Module,
        *,
        device: torch.device,
        process_group: dist.ProcessGroup | None = None,
    ) -> None:
        super().__init__(module, device=device, process_group=process_group)

        params = [param for _name, param in self._managed_params]
        self._multi_grad_hook_handle = torch.autograd.graph.register_multi_grad_hook(
            params, self._on_all_grads_accumulated
        )

    def _on_all_grads_accumulated(
        self, grads: tuple[torch.Tensor | None, ...]
    ) -> None:
        if len(grads) != len(self._managed_params):
            raise RuntimeError(
                "NanoDDPV1: grad hook length mismatch with managed parameters "
                f"({len(grads)} vs {len(self._managed_params)})."
            )
        self.sync_gradients(grads)

    def sync_gradients(
        self, grads: tuple[torch.Tensor | None, ...] | None = None
    ) -> None:
        """Flatten grads, all_reduce, average, write back to ``param.grad``.
        """
        flat = (
            self._flatten_grad_tensors(grads)
            if grads is not None
            else self._flatten_managed_grads()
        )
        dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=self.process_group, async_op=False)
        flat.div_(self._world_size)
        self._unflatten_grads_into_parameters(flat)

    def _flatten_grad_tensors(
        self, grads: tuple[torch.Tensor | None, ...]
    ) -> torch.Tensor:
        chunks: list[torch.Tensor] = []
        for (param_name, _param), grad in zip(self._managed_params, grads):
            if grad is None:
                raise RuntimeError(
                    "NanoDDPV1: expected a gradient for every managed parameter; "
                    f"got None for {param_name!r}."
                )
            chunks.append(grad.detach().reshape(-1))
        return torch.cat(chunks, dim=0)

    def _flatten_managed_grads(self) -> torch.Tensor:
        """Flatten ``param.grad`` into a 1-D tensor (manual ``sync_gradients()``)."""
        chunks: list[torch.Tensor] = []
        for param_name, param in self._managed_params:
            g = param.grad
            if g is None:
                raise RuntimeError(
                    "NanoDDPV1: expected every managed parameter to have .grad; "
                    f"got None for {param_name!r}."
                )
            chunks.append(g.detach().reshape(-1))
        return torch.cat(chunks, dim=0)

    def _unflatten_grads_into_parameters(self, flat: torch.Tensor) -> None:
        """Unflatten grads from a 1-D tensor into the parameters."""
        offset = 0
        for _param_name, param in self._managed_params:
            n = param.numel()
            chunk = flat[offset : offset + n].view_as(param)
            if param.grad is None:
                param.grad = chunk.clone()
            else:
                param.grad.copy_(chunk)
            offset += n


class NanoDDPV2(_NanoDDPBase):
    """V2: all reduce when the grad of each param is ready
    """

    def __init__(
        self,
        module: Module,
        *,
        device: torch.device,
        process_group: dist.ProcessGroup | None = None,
    ) -> None:
        super().__init__(module, device=device, process_group=process_group)
        self._hook_handles: list = []
        # (Work, reduced buffer, param_index, writeback): consumed in ``finalize_backward``.
        self._pending_grad_syncs: list[
            tuple[dist.Work, torch.Tensor, int, bool]
        ] = []

        self._register_per_param_post_accumulate_hooks(
            self._hook_handles, self._on_parameter_grad_ready
        )
        self._register_finalize_when_all_grads_ready(
            self._hook_handles, self.finalize_backward
        )

    def _on_parameter_grad_ready(
        self, param_index: int, grad: torch.Tensor
    ) -> torch.Tensor:
        """Invoked when this parameter's grad has been accumulated for the current backward.
        """
        _, param = self._managed_params[param_index]
        buf = grad.detach().clone()
        work = dist.all_reduce(
            buf,
            op=dist.ReduceOp.SUM,
            group=self.process_group,
            async_op=True,
        )
        self._pending_grad_syncs.append((work, buf, param_index, True))
        return grad

    def finalize_backward(self) -> None:
        """Wait on async collectives; average reduced buffers and write to ``param.grad``.
        """
        for work, _buf, _param_index, _writeback in self._pending_grad_syncs:
            work.wait()
        pending = self._pending_grad_syncs
        self._pending_grad_syncs = []
        for _work, buf, param_index, writeback in pending:
            if not writeback:
                continue
            buf.div_(self._world_size)
            _, param = self._managed_params[param_index]
            if param.grad is None:
                param.grad = buf
            else:
                param.grad.copy_(buf)


_DEFAULT_BUCKET_SIZE_BYTES = 50 * 1024 * 1024


class NanoDDPV3(_NanoDDPBase):
    """V3: similar to V3, but all reduce when the backet is full
    """

    def __init__(
        self,
        module: Module,
        *,
        device: torch.device,
        process_group: dist.ProcessGroup | None = None,
        bucket_size_bytes: int = _DEFAULT_BUCKET_SIZE_BYTES,
    ) -> None:
        if bucket_size_bytes <= 0:
            raise ValueError("bucket_size_bytes must be positive.")
        super().__init__(module, device=device, process_group=process_group)
        self._bucket_size_bytes = bucket_size_bytes
        self._grad_hook_handles: list = []
        # Open bucket: (param_index, detached grad buffer, writeback).
        self._open_bucket: list[tuple[int, torch.Tensor, bool]] = []
        self._open_bucket_nbytes: int = 0
        # (Work, flat reduced buffer, unflatten metadata).
        self._pending_bucket_syncs: list[
            tuple[dist.Work, torch.Tensor, list[tuple[int, int, int, bool]]]
        ] = []

        self._register_per_param_post_accumulate_hooks(
            self._grad_hook_handles, self._on_parameter_grad_ready
        )
        self._register_finalize_when_all_grads_ready(
            self._grad_hook_handles, self.finalize_backward
        )

    def _on_parameter_grad_ready(
        self, param_index: int, grad: torch.Tensor
    ) -> torch.Tensor:
        _, param = self._managed_params[param_index]
        buf = grad.detach().clone()
        writeback = True

        param_nbytes = buf.numel() * buf.element_size()
        if (
            self._open_bucket
            and self._open_bucket_nbytes + param_nbytes > self._bucket_size_bytes
        ):
            self._flush_open_bucket()

        self._open_bucket.append((param_index, buf, writeback))
        self._open_bucket_nbytes += param_nbytes
        return grad

    def _flush_open_bucket(self) -> None:
        if not self._open_bucket:
            return

        flat = torch.cat(
            [buf.reshape(-1) for _param_index, buf, _writeback in self._open_bucket],
            dim=0,
        )
        work = dist.all_reduce(
            flat,
            op=dist.ReduceOp.SUM,
            group=self.process_group,
            async_op=True,
        )
        meta: list[tuple[int, int, int, bool]] = []
        offset = 0
        for param_index, buf, writeback in self._open_bucket:
            n = buf.numel()
            meta.append((param_index, offset, n, writeback))
            offset += n
        self._pending_bucket_syncs.append((work, flat, meta))
        self._open_bucket = []
        self._open_bucket_nbytes = 0

    def finalize_backward(self) -> None:
        """Flush tail bucket, wait on bucket collectives, average and write ``param.grad``."""
        self._flush_open_bucket()
        for work, _flat, _meta in self._pending_bucket_syncs:
            work.wait()
        pending = self._pending_bucket_syncs
        self._pending_bucket_syncs = []
        for _work, flat, meta in pending:
            flat.div_(self._world_size)
            for param_index, offset, n, writeback in meta:
                if not writeback:
                    continue
                chunk = flat[offset : offset + n]
                _, param = self._managed_params[param_index]
                averaged = chunk.view_as(param)
                if param.grad is None:
                    param.grad = averaged.clone()
                else:
                    param.grad.copy_(averaged)

