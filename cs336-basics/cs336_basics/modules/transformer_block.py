from torch import nn
import torch

from cs336_basics.modules import RMSNorm, CausalMultiHeadSelfAttention, FFN


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_heads: int,
        max_seq_len: int,
        activation: str,
        with_rope: bool,
        theta: float | None,
        with_norm: bool,
        pre_norm: bool,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super(TransformerBlock, self).__init__()
        self.with_norm = with_norm
        self.pre_norm = pre_norm
        self.with_rope = with_rope

        if with_norm:
            self.norm1 = RMSNorm(d_model=d_model, eps=eps, device=device, dtype=dtype)
            self.norm2 = RMSNorm(d_model=d_model, eps=eps, device=device, dtype=dtype)

        self.attn = CausalMultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            with_rope=with_rope,
            theta=theta,
            device=device,
            dtype=dtype
        )
        
        self.ffn = FFN(
            d_model=d_model,
            d_ff=d_ff,
            activation=activation,
            device=device,
            dtype=dtype
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        _, s, _ = x.shape
        token_positions = torch.arange(0, s, step=1, device=x.device, dtype=torch.int) \
            if self.with_rope else None

        if self.with_norm and self.pre_norm:
            x = x + self.attn(self.norm1(x), token_positions, attn_mask)
            x = x + self.ffn(self.norm2(x))
        elif self.with_norm:
            x = self.norm1(x + self.attn(x, token_positions, attn_mask))
            x = self.norm2(x + self.ffn(x))
        else:
            x = x + self.attn(x, token_positions, attn_mask)
            x = x + self.ffn(x)

        return x
