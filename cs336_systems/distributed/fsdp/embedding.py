import torch
import torch.distributed as dist
from torch import nn

from .common import ShardInfo


EMBEDDING_WEIGHT_ATTRIBUTE = "embeddings"


def shard_embedding(fsdp, module: nn.Module) -> None:
    full_parameter = getattr(module, EMBEDDING_WEIGHT_ATTRIBUTE)
    full_data = full_parameter.detach().float()
    vocab_size, d_model = full_data.shape
    rows_per_rank = (vocab_size + fsdp.world_size - 1) // fsdp.world_size

    padded = torch.zeros(
        rows_per_rank * fsdp.world_size,
        d_model,
        dtype=torch.float32,
        device=full_data.device,
    )
    padded[:vocab_size] = full_data
    shard_data = padded[fsdp.rank * rows_per_rank: (fsdp.rank + 1) * rows_per_rank].clone()
    shard_parameter = nn.Parameter(shard_data, requires_grad=full_parameter.requires_grad)
    setattr(module, EMBEDDING_WEIGHT_ATTRIBUTE, shard_parameter)

    info = ShardInfo(
        parameter=shard_parameter,
        full_shape=full_parameter.shape,
        full_numel=full_parameter.numel(),
        shard_numel=shard_data.numel(),
        shard_data=shard_data,
        kind="embedding",
    )
    fsdp._shards.append(info)
    fsdp._shard_by_id[id(shard_parameter)] = info

    module.register_forward_pre_hook(lambda module, inputs, info=info: _gather_weight(fsdp, info))
    module.register_forward_hook(
        lambda module, inputs, output, info=info: _free_weight_and_register_backward_hook(fsdp, info, output)
    )
    shard_parameter.register_post_accumulate_grad_hook(lambda parameter, info=info: _reduce_gradient(fsdp, info))


def _gather_weight(fsdp, info: ShardInfo) -> None:
    gathered = [torch.empty_like(info.shard_data) for _ in range(fsdp.world_size)]
    dist.all_gather(gathered, info.shard_data)
    full = torch.cat(gathered, dim=0)[:info.full_shape[0]]
    if fsdp.compute_dtype is not None:
        full = full.to(fsdp.compute_dtype)
    info.parameter.data = full


def _free_weight_and_register_backward_hook(fsdp, info: ShardInfo, output: torch.Tensor) -> None:
    info.parameter.data = info.shard_data
    if output.requires_grad:
        output.register_hook(lambda grad: _gather_for_backward(fsdp, info, grad))


def _gather_for_backward(fsdp, info: ShardInfo, grad: torch.Tensor) -> torch.Tensor:
    _gather_weight(fsdp, info)
    return grad


def _reduce_gradient(fsdp, info: ShardInfo) -> None:
    gradient = info.parameter.grad
    if gradient is None:
        return

    full_gradient = gradient.reshape(-1).float()
    padded_numel = info.shard_numel * fsdp.world_size
    if full_gradient.numel() < padded_numel:
        padded = torch.zeros(padded_numel, dtype=torch.float32, device=full_gradient.device)
        padded[:full_gradient.numel()] = full_gradient
        full_gradient = padded

    shard_gradient = torch.empty(
        info.shard_numel,
        dtype=torch.float32,
        device=full_gradient.device,
    )
    dist.reduce_scatter_tensor(shard_gradient, full_gradient, op=dist.ReduceOp.SUM)
    shard_gradient.div_(fsdp.world_size)
    info.parameter.data = info.shard_data
    info.parameter.grad = shard_gradient.reshape(info.shard_data.shape)
