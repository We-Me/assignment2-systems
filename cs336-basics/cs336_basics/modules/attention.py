import math

import torch
from torch import nn
from einops import einsum, rearrange

from cs336_basics.functions import softmax
from cs336_basics.modules import Linear, RoPE


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor
):
    scale = math.pow(Q.shape[-1], -0.5)
    attn = einsum(Q, K, "... s_q d_k, ... s_k d_k -> ... s_q s_k")
    attn = torch.mul(attn, scale)
    attn = attn.masked_fill_(~mask, float("-inf"))
    attn = softmax(attn, dim=-1)
    output = einsum(attn, V, "... s_q s_k, ... s_k d_v -> ... s_q d_v")
    return output


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        with_rope: bool,
        theta: float | None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super(CausalMultiHeadSelfAttention, self).__init__()
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = Linear(d_model, d_model, device, dtype)
        self.k_proj = Linear(d_model, d_model, device, dtype)
        self.v_proj = Linear(d_model, d_model, device, dtype)
        self.o_proj = Linear(d_model, d_model, device, dtype)

        if with_rope:
            if max_seq_len is None or theta is None:
                raise ValueError("max_seq_len and theta is need as with_rope")
            self.rope = RoPE(theta=theta, d_k=self.head_dim, max_seq_len=max_seq_len, device=device)
        self.with_rope = with_rope

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _, s, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = rearrange(q, "b s (h d) -> (b h) s d", h=self.num_heads, d=self.head_dim)
        k = rearrange(k, "b s (h d) -> (b h) s d", h=self.num_heads, d=self.head_dim)
        v = rearrange(v, "b s (h d) -> (b h) s d", h=self.num_heads, d=self.head_dim)

        if self.with_rope:
            if token_positions is None:
                raise RuntimeError("token_positions is needed as with_rope")
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        causal_mask = torch.tril(torch.ones(s, s, device=x.device, dtype=x.dtype), diagonal=0).bool()
        mask = causal_mask if attn_mask is None else causal_mask & attn_mask

        output = scaled_dot_product_attention(q, k, v, mask)
        output = rearrange(output, "(b h) s d -> b s (h d)", h=self.num_heads, d=self.head_dim)
        output = self.o_proj(output)
        return output
