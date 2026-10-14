import math

import torch
import torch.cuda.nvtx as nvtx
from einops import einsum

from cs336_basics.functions import softmax


@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor
):
    scale = math.pow(Q.shape[-1], -0.5)
    with nvtx.range("computing attention scores"):
        attn = einsum(Q, K, "... s_q d_k, ... s_k d_k -> ... s_q s_k")
        attn = torch.mul(attn, scale)
        attn = attn.masked_fill_(~mask, float("-inf"))

    with nvtx.range("computing softmax"):
        attn = softmax(attn, dim=-1)

    with nvtx.range("final matmul"):
        output = einsum(attn, V, "... s_q s_k, ... s_k d_v -> ... s_q d_v")
    return output
