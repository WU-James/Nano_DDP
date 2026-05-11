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
#   NanoDDPPathA — post-backward: flatten -> all_reduce -> unflatten (call sync_gradients()).
#   NanoDDPPathB — backward hooks on AccumulateGrad (closer to official DDP).
#   NanoDDP — umbrella: both path stubs in one file; implement only one strategy.
#
# Path A training loop (NanoDDPPathA registers register_full_backward_hook → auto sync_gradients):
#   model = NanoDDPPathA(raw_model, device=...)
#   loss.backward()   # ends with in-place sync all_reduce over flattened grads; no finalize_backward needed
#   optimizer.step()
#   Call model.sync_gradients() manually only if you disable the hook (not default).
#
# Path B: implement hooks, then optimizer.step() after backward (+ finalize if async).

from __future__ import annotations

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


class NanoDDPPathA(_NanoDDPBase):
    """Path A: after full backward on ``module``, flatten → sync all_reduce(SUM) → / world_size → unflatten.

    Uses ``register_full_backward_hook`` so ``sync_gradients()`` runs automatically; no ``finalize_backward`` needed
    for the default synchronous ``dist.all_reduce`` (async_op=False).
    """

    def __init__(
        self,
        module: Module,
        *,
        device: torch.device,
        process_group: dist.ProcessGroup | None = None,
    ) -> None:
        super().__init__(module, device=device, process_group=process_group)

        # Register full backward hook to trigger sync_gradients after backward.
        self._full_backward_hook_handle = self.module.register_full_backward_hook(
            self._post_backward_sync
        )

    def _post_backward_sync(
        self,
        module: Module,
        grad_input: tuple[torch.Tensor | None, ...],
        grad_output: tuple[torch.Tensor | None, ...],
    ) -> None:
        del module, grad_input, grad_output
        self.sync_gradients()

    def sync_gradients(self) -> None:
        """Flatten managed grads, all_reduce, write back to ``param.grad``."""
        flat = self._flatten_managed_grads()
        dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=self.process_group, async_op=False)
        flat.div_(self._world_size)
        self._unflatten_grads_into_parameters(flat)

    def _flatten_managed_grads(self) -> torch.Tensor:
        """Flatten managed grads into a 1-D tensor."""
        chunks: list[torch.Tensor] = []
        for param_name, param in self._managed_params:
            g = param.grad
            if g is None:
                raise RuntimeError(
                    "NanoDDPPathA: expected every managed parameter to have .grad after backward; "
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
            param.grad.copy_(chunk)
            offset += n


class NanoDDPPathB(_NanoDDPBase):
    """Path B only: register_gradient_accumulator_hooks; all_reduce during backward."""

    def __init__(
        self,
        module: Module,
        *,
        device: torch.device,
        process_group: dist.ProcessGroup | None = None,
    ) -> None:
        super().__init__(module, device=device, process_group=process_group)
        self._grad_hook_handles: list = []
        # Append ``Work`` from ``dist.all_reduce(..., async_op=True)``; consumed in ``finalize_backward``.
        self._pending_works: list = []

        # TODO: after implementing _register_gradient_accumulator_hooks, uncomment:
        # self._register_gradient_accumulator_hooks()

    def _register_gradient_accumulator_hooks(self) -> None:
        """TODO: For each index idx and (param_name, p) in self._managed_params:

        tmp = p.expand_as(p)
        grad_acc = tmp.grad_fn.next_functions[0][0]
        h = grad_acc.register_hook(functools.partial(self._on_gradient_accumulator_ready, idx))

        Append h to self._grad_hook_handles. Import functools in this file when you fill this in.

        Do not also run Path A sync_gradients on the same step.
        """
        ...

    def _on_gradient_accumulator_ready(
        self, param_index: int, grad: torch.Tensor | None
    ) -> torch.Tensor | None:
        """TODO: Invoked when this parameter's grad has been computed for the current backward.

        Typical: dist.all_reduce(grad, ...); grad.div_(world_size); return grad.

        grad may be None for unused parameters on this step — align collectives across ranks or restrict models.

        Return None to leave .grad as computed; return a Tensor to replace the gradient for this parameter.
        """
        ...

    def finalize_backward(self) -> None:
        """Wait on async collectives in ``_pending_works``; call after ``backward()`` if hooks use ``async_op=True``."""
        for work in self._pending_works:
            work.wait()
        self._pending_works.clear()


# class NanoDDP(_NanoDDPBase):
#     """Umbrella teaching class: Path A + Path B methods together; fill in only one path.

#     Same layout as before the split — convenient for a single file of TODOs.
#     """

#     def __init__(
#         self,
#         module: Module,
#         *,
#         device: torch.device,
#         process_group: dist.ProcessGroup | None = None,
#     ) -> None:
#         super().__init__(module, device=device, process_group=process_group)
#         self._grad_hook_handles: list = []
#         self._pending_works: list = []

#         # Path B (optional): implement _register_gradient_accumulator_hooks first, then uncomment:
#         # self._register_gradient_accumulator_hooks()

#     # ----- Path A (post-backward flat sync) -----

#     def sync_gradients(self) -> None:
#         """TODO: flat = _flatten_managed_grads(); _all_reduce_maybe_async(flat);
#         after reduce completes: divide flat by world_size and write back into each p.grad
#         (_unflatten_grads_into_parameters), unless you folded divide/write into helpers.

#         For fully synchronous dist.all_reduce, you can skip _pending_works and finalize_backward.
#         If you implement Path B hooks with all_reduce, leave this as no-op or only wait().
#         """
#         ...

#     def _flatten_managed_grads(self) -> torch.Tensor:
#         """TODO: One 1-D tensor containing all managed gradients (e.g. torch.cat). Handle None grads."""
#         ...

#     def _all_reduce_maybe_async(self, tensor: torch.Tensor) -> None:
#         """TODO: dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=..., async_op=...).

#         If async_op=True, append returned Work to self._pending_works.
#         """
#         ...

#     def _unflatten_grads_into_parameters(self, flat: torch.Tensor) -> None:
#         """TODO: After flat has been summed across ranks and divided by world_size, scatter slices back to param.grad."""
#         ...

#     def finalize_backward(self) -> None:
#         """Path B / async: ``work.wait()`` on ``_pending_works``. Path A-only users can ignore."""
#         for work in self._pending_works:
#             work.wait()
#         self._pending_works.clear()

#     # ----- Path B (backward hooks) -----

#     def _register_gradient_accumulator_hooks(self) -> None:
#         """TODO: For each index idx and (param_name, p) in self._managed_params:

#         tmp = p.expand_as(p)
#         grad_acc = tmp.grad_fn.next_functions[0][0]
#         h = grad_acc.register_hook(functools.partial(self._on_gradient_accumulator_ready, idx))

#         Append h to self._grad_hook_handles. Import functools in this file when you fill this in.

#         Do not double-reduce: if you hook-drive all_reduce, leave sync_gradients() as no-op or only wait().
#         """
#         ...

#     def _on_gradient_accumulator_ready(
#         self, param_index: int, grad: torch.Tensor | None
#     ) -> torch.Tensor | None:
#         """TODO: Invoked when this parameter's grad has been computed for the current backward.

#         grad may be None for unused parameters on this step — align collectives across ranks or restrict models.

#         Return None to leave .grad as computed; return a Tensor to replace the gradient for this parameter.
#         """
#         ...


# def setup_process_group_stub() -> None:
#     """TODO (outside class): init_process_group, set_device — keep in your train script."""
#     ...
