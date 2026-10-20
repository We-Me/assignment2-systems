from collections.abc import Callable

import torch
import torch.distributed as dist
from torch import nn


class OverlapDDP(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("OverlapDDP requires an initialized torch.distributed process group")

        self.module = module
        self.world_size = dist.get_world_size()
        self._pending_handles: list[object] = []
        self._broadcast_parameters()
        self._register_gradient_hooks()

    def _broadcast_parameters(self) -> None:
        for parameter in self.module.parameters():
            dist.broadcast(parameter.data, src=0)

    def _register_gradient_hooks(self) -> None:
        for parameter in self.module.parameters():
            if parameter.requires_grad:
                parameter.register_post_accumulate_grad_hook(self._make_gradient_hook(parameter))

    def _make_gradient_hook(self, parameter: nn.Parameter) -> Callable[[torch.Tensor], None]:
        def hook(_: torch.Tensor) -> None:
            gradient = parameter.grad
            if gradient is None:
                return
            if gradient.is_sparse:
                raise RuntimeError("OverlapDDP does not support sparse gradients")
            if not gradient.is_contiguous():
                gradient = gradient.contiguous()
                parameter.grad = gradient

            handle = dist.all_reduce(gradient, op=dist.ReduceOp.SUM, async_op=True)
            self._pending_handles.append(handle)

        return hook

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        if not self._pending_handles:
            return

        for handle in self._pending_handles:
            handle.wait()

        for parameter in self.module.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(self.world_size)

        self._pending_handles.clear()
