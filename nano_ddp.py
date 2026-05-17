# Teaching skeleton: single-machine multi-process DDP-style wrapper (gradients only).
# Fill in the bodies marked TODO. This copy lives in nano_ddp/ (standalone from PyTorch agent_space).
#
# Prerequisites (outside this file):
#   torch.distributed.init_process_group(...)
#   torch.cuda.set_device(local_rank)  # per process
#   Place module on self.device before wrapping (same as official DDP).
#   No BatchNorm / module buffer sync in this skeleton (use SyncBatchNorm outside if needed).
#
# Classes:
#   NanoDDPV1 — post-backward: flatten -> all_reduce -> unflatten (call sync_gradients()).
#   NanoDDPV2 — backward hooks on AccumulateGrad (closer to official DDP).
#   NanoDDPV3 — V2 + gradient buckets: all_reduce only when a bucket is full (default 25 MiB).
#   NanoDDP — umbrella: both path stubs in one file; implement only one strategy.
#
# V1 training loop (register_multi_grad_hook on managed params → auto sync_gradients):
#   model = NanoDDPV1(raw_model, device=...)
#   loss.backward()   # ends with in-place sync all_reduce over flattened grads; no finalize_backward needed
#   optimizer.step()
#   Call model.sync_gradients() manually only if you disable the hook (not default).
#
# V2 training loop (register_full_backward_hook → auto finalize_backward):
#   model = NanoDDPV2(raw_model, device=...)
#   loss.backward()   # per-param async all_reduce during backward; hook waits and averages
#   optimizer.step()
#   Call model.finalize_backward() manually only if you disable the hook (not default).
#
# V3 training loop (same as V2, but grads are bucketed before all_reduce):
#   model = NanoDDPV3(raw_model, device=..., bucket_size_bytes=25 * 1024 * 1024)
#   loss.backward()   # flush full buckets during backward; finalize flushes the tail bucket
#   optimizer.step()

from __future__ import annotations

import functools

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

    # def _allreduce_unused_params_placeholder(self) -> None:
    #     """TODO: If some ranks omit parameters from the graph, plain all_reduce can deadlock.

    #     Restrict models or add zeros/masks like official Reducer.
    #     """
    #     ...

    # def _assert_same_control_flow(self) -> None:
    #     """TODO: Debug: collective order matches on every rank."""
    #     ...


class NanoDDPV1(_NanoDDPBase):
    """V1: after all parameter grads are ready, flatten → sync all_reduce(SUM) → / world_size → unflatten.

    Uses ``torch.autograd.graph.register_multi_grad_hook`` on managed parameters so
    ``sync_gradients()`` runs after backward even when the inner module returns a
    Hugging Face ``ModelOutput`` (``register_full_backward_hook`` would not fire).
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

        When called from ``register_multi_grad_hook``, pass the hook's ``grads`` tuple:
        ``param.grad`` may not be populated yet at callback time.
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
    """V2: per-parameter AccumulateGrad hooks (async all_reduce) + ``finalize_backward`` after backward.

    Uses ``register_full_backward_hook`` so ``finalize_backward()`` runs automatically; manual call is optional.
    """

    def __init__(
        self,
        module: Module,
        *,
        device: torch.device,
        process_group: dist.ProcessGroup | None = None,
    ) -> None:
        super().__init__(module, device=device, process_group=process_group)
        self._grad_hook_handles: list = []
        # (Work, reduced buffer, param_index, writeback): consumed in ``finalize_backward``.
        self._pending_grad_syncs: list[
            tuple[dist.Work, torch.Tensor, int, bool]
        ] = []

        self._register_gradient_accumulator_hooks()
        self._full_backward_hook_handle = self.module.register_full_backward_hook(
            self._post_backward_finalize
        )

    def _post_backward_finalize(
        self,
        module: Module,
        grad_input: tuple[torch.Tensor | None, ...],
        grad_output: tuple[torch.Tensor | None, ...],
    ) -> None:
        del module, grad_input, grad_output
        self.finalize_backward()

    def _register_gradient_accumulator_hooks(self) -> None:
        """For each managed parameter, hook its AccumulateGrad node.
        """
        for idx, (_param_name, param) in enumerate(self._managed_params):
            tmp = param.expand_as(param)
            grad_acc = tmp.grad_fn.next_functions[0][0]
            h = grad_acc.register_hook(
                functools.partial(self._on_gradient_accumulator_ready, idx)
            )
            self._grad_hook_handles.append(h)

    def _on_gradient_accumulator_ready(
        self, param_index: int, grad: torch.Tensor | None
    ) -> torch.Tensor | None:
        """Invoked when this parameter's grad has been computed for the current backward.

        grad may be None for unused parameters on this step — run a zero all_reduce to
        align collectives across ranks, then return None.

        Reduce runs on a detached buffer so autograd does not traverse c10d collectives.
        """
        _, param = self._managed_params[param_index]
        if grad is None:
            buf = torch.zeros_like(param)
            writeback = False
        else:
            buf = grad.detach().clone()
            writeback = True

        work = dist.all_reduce(
            buf,
            op=dist.ReduceOp.SUM,
            group=self.process_group,
            async_op=True,
        )
        self._pending_grad_syncs.append((work, buf, param_index, writeback))
        return grad

    def finalize_backward(self) -> None:
        """Wait on async collectives; average reduced buffers and write to ``param.grad``.

        Called automatically via full backward hook; safe to call again (no-op if nothing pending).
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


_DEFAULT_BUCKET_SIZE_BYTES = 25 * 1024 * 1024


class NanoDDPV3(_NanoDDPBase):
    """V3: V2-style AccumulateGrad hooks, but batch grads into byte-limited buckets before all_reduce.

    Parameters are appended to the open bucket as their grads become ready during backward.
    When the next parameter would exceed ``bucket_size_bytes``, the current bucket is flattened,
    all_reduced asynchronously, and a new bucket starts. ``finalize_backward`` flushes any
    remainder and writes averaged grads back to parameters.
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

        self._register_gradient_accumulator_hooks()
        self._full_backward_hook_handle = self.module.register_full_backward_hook(
            self._post_backward_finalize
        )

    def _post_backward_finalize(
        self,
        module: Module,
        grad_input: tuple[torch.Tensor | None, ...],
        grad_output: tuple[torch.Tensor | None, ...],
    ) -> None:
        del module, grad_input, grad_output
        self.finalize_backward()

    def _register_gradient_accumulator_hooks(self) -> None:
        for idx, (_param_name, param) in enumerate(self._managed_params):
            tmp = param.expand_as(param)
            grad_acc = tmp.grad_fn.next_functions[0][0]
            h = grad_acc.register_hook(
                functools.partial(self._on_gradient_accumulator_ready, idx)
            )
            self._grad_hook_handles.append(h)

    def _on_gradient_accumulator_ready(
        self, param_index: int, grad: torch.Tensor | None
    ) -> torch.Tensor | None:
        _, param = self._managed_params[param_index]
        if grad is None:
            buf = torch.zeros_like(param)
            writeback = False
        else:
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

