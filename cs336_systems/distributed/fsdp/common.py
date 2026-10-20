from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn


@dataclass
class ShardInfo:
    parameter: nn.Parameter
    full_shape: torch.Size
    full_numel: int
    shard_numel: int
    shard_data: torch.Tensor
    kind: str
    index: int
    gather_handle: object | None = None
    gather_outputs: list[torch.Tensor] | None = None
    gathered: bool = False
    pending_grad: torch.Tensor | None = None


def start_gather(fsdp, info: ShardInfo) -> None:
    if info.gathered or info.gather_handle is not None:
        return

    outputs = [torch.empty_like(info.shard_data) for _ in range(fsdp.world_size)]
    info.gather_outputs = outputs
    info.gather_handle = dist.all_gather(outputs, info.shard_data, async_op=True)


def wait_gather(fsdp, info: ShardInfo) -> None:
    if info.gathered:
        return
    if info.gather_handle is None:
        start_gather(fsdp, info)

    info.gather_handle.wait()
    outputs = info.gather_outputs
    if info.kind == "embedding":
        full = torch.cat(outputs, dim=0)[:info.full_shape[0]]
    else:
        full = torch.cat(outputs)[:info.full_numel].reshape(info.full_shape)
    if fsdp.compute_dtype is not None:
        full = full.to(fsdp.compute_dtype)

    info.parameter.data = full
    info.gathered = True
    info.gather_handle = None
    info.gather_outputs = None


def free_weight(info: ShardInfo) -> None:
    info.parameter.data = info.shard_data
    info.gathered = False
