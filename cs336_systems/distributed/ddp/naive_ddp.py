import torch
import torch.distributed as dist
from torch import nn


class NaiveDDP(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("NaiveDDP requires an initialized torch.distributed process group")

        self.module = module
        self.world_size = dist.get_world_size()
        self._broadcast_parameters()

    def _broadcast_parameters(self) -> None:
        for parameter in self.module.parameters():
            dist.broadcast(parameter.data, src=0)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        for parameter in self.module.parameters():
            gradient = parameter.grad
            if gradient is None:
                continue
            if gradient.is_sparse:
                raise RuntimeError("NaiveDDP does not support sparse gradients")
            if not gradient.is_contiguous():
                gradient = gradient.contiguous()
                parameter.grad = gradient
            dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
            gradient.div_(self.world_size)
