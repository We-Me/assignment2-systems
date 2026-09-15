import torch
from torch import nn

from cs336_basics.modules import Linear
from cs336_basics.functions import silu


SUPPORT_ACTIVATION = {
    "silu": silu,
    "swiglu": None,
}


class FFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        activation: str,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super(FFN, self).__init__()
        assert activation.lower() in SUPPORT_ACTIVATION
        self.activation = activation.lower()

        self.linear1 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        self.linear2 = Linear(in_features=d_ff, out_features=d_model, device=device, dtype=dtype)
        if self.activation == "swiglu":
            self.linear3 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "swiglu":
            gate = silu(self.linear1(x))
            x = self.linear3(x)
            return self.linear2(gate * x)
        else:
            return self.linear2(SUPPORT_ACTIVATION[self.activation](self.linear1(x)))
