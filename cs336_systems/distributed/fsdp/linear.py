import torch
import torch.distributed as dist
from torch import nn

from .common import ShardInfo, free_weight, wait_gather


LINEAR_WEIGHT_ATTRIBUTE = "weights"


def shard_linear(fsdp, module: nn.Module) -> None:
    full_parameter = getattr(module, LINEAR_WEIGHT_ATTRIBUTE)
    full_data = full_parameter.detach().reshape(-1).float()
    full_numel = full_data.numel()
    shard_numel = (full_numel + fsdp.world_size - 1) // fsdp.world_size

    padded = torch.zeros(
        shard_numel * fsdp.world_size,
        dtype=torch.float32,
        device=full_data.device,
    )
    padded[:full_numel] = full_data
    shard_data = padded[fsdp.rank * shard_numel: (fsdp.rank + 1) * shard_numel].clone()
    shard_parameter = nn.Parameter(shard_data, requires_grad=full_parameter.requires_grad)
    setattr(module, LINEAR_WEIGHT_ATTRIBUTE, shard_parameter)

    info = ShardInfo(
        parameter=shard_parameter,
        full_shape=full_parameter.shape,
        full_numel=full_numel,
        shard_numel=shard_numel,
        shard_data=shard_data,
        kind="linear",
        index=len(fsdp._shards),
    )
    fsdp._shards.append(info)
    fsdp._shard_by_id[id(shard_parameter)] = info

    module.register_forward_pre_hook(lambda module, inputs, info=info: wait_gather(fsdp, info))
    module.register_forward_hook(lambda module, inputs, output, info=info: _after_forward(fsdp, info))
    module.register_full_backward_pre_hook(lambda module, grad_output, info=info: wait_gather(fsdp, info))
    module.register_full_backward_hook(
        lambda module, grad_input, grad_output, info=info: _after_backward(fsdp, info)
    )
    shard_parameter.register_post_accumulate_grad_hook(
        lambda parameter, info=info: _reduce_gradient(fsdp, info)
    )


def _after_forward(fsdp, info: ShardInfo) -> None:
    free_weight(info)
    fsdp._prefetch(info.index + 2)


def _after_backward(fsdp, info: ShardInfo) -> None:
    free_weight(info)
    fsdp._prefetch(info.index - 1)


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
    info.pending_grad = shard_gradient
