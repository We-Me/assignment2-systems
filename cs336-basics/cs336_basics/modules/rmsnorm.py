import torch
from torch import nn
from einops import einsum


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super(RMSNorm, self).__init__()
        self.g = nn.Parameter(
            data=torch.empty(size=(d_model,), device=device, dtype=dtype),
            requires_grad=True,
        )
        self.eps = eps

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.ones_(self.g)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(input=(torch.mean(x.square(), dim=-1, keepdim=True) + self.eps))
        result = einsum(x, self.g, "batch sequence d_model, d_model -> batch sequence d_model")

        result = torch.divide(result, rms)
        return result.to(in_dtype)
