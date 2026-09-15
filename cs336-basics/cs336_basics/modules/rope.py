import torch
from torch import nn

from einops import einsum, rearrange


class RoPE(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ) -> None:
        super(RoPE, self).__init__()
        assert d_k % 2 == 0

        rot_matrix = self.create(theta, d_k, max_seq_len, device)
        self.register_buffer(name="rot_matrix", tensor=rot_matrix, persistent=False)

    @staticmethod
    def create(theta: float, d_k: int, max_seq_len: int, device: torch.device | None):
        r_k2 = torch.arange(start=0, end=d_k, step=2, device=device, dtype=torch.float32).unsqueeze_(0)
        r_i = torch.arange(start=0, end=max_seq_len, step=1, device=device, dtype=torch.float32).unsqueeze_(1)
        theta_ik = r_i / (theta ** (r_k2 / d_k))
        theta_ik = rearrange(theta_ik, "max_seq half_d -> max_seq half_d 1 1")

        sin_theta = torch.sin(theta_ik)
        cos_theta = torch.cos(theta_ik)
        sin_neg_theta = torch.sin(theta_ik.neg())

        cos_sin_neg = torch.cat((cos_theta, sin_neg_theta), dim=3)
        sin_cos = torch.cat((sin_theta, cos_theta), dim=3)

        rot_matrix = torch.cat((cos_sin_neg, sin_cos), dim=2)
        return rot_matrix

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        rot = self.rot_matrix[token_positions]
        if rot.dim() == 4:
            rot = rearrange(rot, "s h p1 p2 -> 1 s h p1 p2")
            rot.expand(x.shape[0], *rot.shape[1:])

        x = rearrange(x, "b s (half_d pair) -> b s half_d pair", pair=2)
        x = einsum(rot, x, "b s h p1 p2, b s h p2 -> b s h p1")
        x = rearrange(x, "b s half_d pair -> b s (half_d pair)")
        return x
