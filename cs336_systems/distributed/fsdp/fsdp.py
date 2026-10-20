import torch
import torch.distributed as dist
from torch import nn

from cs336_basics.modules import Embedding, Linear

from .embedding import shard_embedding
from .linear import shard_linear
from .common import ShardInfo


class FSDP(nn.Module):
    def __init__(self, module: nn.Module, compute_dtype: torch.dtype | None = None) -> None:
        super().__init__()
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("FSDP requires an initialized torch.distributed process group")

        self.module = module
        self.compute_dtype = compute_dtype
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self._shards: list[ShardInfo] = []
        self._shard_by_id: dict[int, ShardInfo] = {}

        for submodule in module.modules():
            if isinstance(submodule, Linear):
                shard_linear(self, submodule)
            elif isinstance(submodule, Embedding):
                shard_embedding(self, submodule)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        for info in self._shards:
            info.parameter.data = info.shard_data
            if info.pending_grad is not None:
                info.parameter.grad = info.pending_grad
                info.pending_grad = None

        for parameter in self.module.parameters():
            if id(parameter) in self._shard_by_id:
                continue
            if parameter.grad is None:
                continue
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            parameter.grad.div_(self.world_size)

    def gather_full_params(self) -> dict[str, torch.Tensor]:
        full_params = {}
        for name, parameter in self.module.named_parameters():
            info = self._shard_by_id.get(id(parameter))
            if info is None:
                full_params[name] = parameter.data
                continue

            gathered = [torch.empty_like(info.shard_data) for _ in range(self.world_size)]
            dist.all_gather(gathered, info.shard_data)
            if info.kind == "embedding":
                full_params[name] = torch.cat(gathered, dim=0)[:info.full_shape[0]]
            else:
                full_params[name] = torch.cat(gathered)[:info.full_numel].reshape(info.full_shape)

        return full_params
