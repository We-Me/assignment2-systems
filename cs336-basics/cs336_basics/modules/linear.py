import math

import torch
from torch import nn
from einops import einsum


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super(Linear, self).__init__()
        self.weights = nn.Parameter(
            data=torch.empty(size=(out_features, in_features), device=device, dtype=dtype),
            requires_grad=True,
        )

        self._reset_parameters()

    def _reset_parameters(self):
        std = math.sqrt(2.0 / (self.weights.size(0) + self.weights.size(1)))
        nn.init.trunc_normal_(self.weights, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weights, "batch sequence d_in, d_out d_in -> batch sequence d_out")
