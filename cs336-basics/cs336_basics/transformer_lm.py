import torch
from torch import nn

from cs336_basics.modules import Embedding, TransformerBlock, RMSNorm, Linear


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        d_ff: int,
        num_heads: int,
        activation: str = "swiglu",
        with_rope: bool = True,
        theta: float | None = 10000.0,
        with_norm: bool = True,
        pre_norm: bool = True,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super(TransformerLM, self).__init__()
        self.token_embeddings = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
        )

        layers = [
            TransformerBlock(
                d_model=d_model,
                d_ff=d_ff,
                num_heads=num_heads,
                max_seq_len=context_length,
                activation=activation,
                with_rope=with_rope,
                theta=theta,
                with_norm=with_norm,
                pre_norm=pre_norm,
                eps=eps,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ]
        self.layers = nn.Sequential(*layers)

        self.with_norm = with_norm
        if with_norm:
            self.norm = RMSNorm(
                d_model=d_model,
                eps=eps,
                device=device,
                dtype=dtype,
            )

        self.lm_head = Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device,
            dtype=dtype,
        )

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(in_indices)
        x = self.layers(x)
        if self.with_norm:
            x = self.norm(x)
        x = self.lm_head(x)
        return x
