import torch
import torch.distributed as dist
from torch import nn


class FlatDDP(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("FlatDDP requires an initialized torch.distributed process group")

        self.module = module
        self.world_size = dist.get_world_size()
        self._broadcast_parameters()

    def _broadcast_parameters(self) -> None:
        for parameter in self.module.parameters():
            dist.broadcast(parameter.data, src=0)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        gradients: list[torch.Tensor] = []
        for parameter in self.module.parameters():
            gradient = parameter.grad
            if gradient is None:
                continue
            if gradient.is_sparse:
                raise RuntimeError("FlatDDP does not support sparse gradients")
            gradients.append(gradient)

        if not gradients:
            return

        flat_gradient = torch.cat([gradient.reshape(-1) for gradient in gradients])
        dist.all_reduce(flat_gradient, op=dist.ReduceOp.SUM)
        flat_gradient.div_(self.world_size)

        offset = 0
        for gradient in gradients:
            numel = gradient.numel()
            gradient.copy_(flat_gradient[offset: offset + numel].view_as(gradient))
            offset += numel
