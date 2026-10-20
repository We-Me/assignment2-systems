from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ShardInfo:
    parameter: nn.Parameter
    full_shape: torch.Size
    full_numel: int
    shard_numel: int
    shard_data: torch.Tensor
    kind: str
    pending_grad: torch.Tensor | None = None
