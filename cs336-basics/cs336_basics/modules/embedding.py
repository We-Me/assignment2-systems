import torch
from torch import nn


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super(Embedding, self).__init__()
        self.embeddings = nn.Parameter(
            data=torch.empty(size=(num_embeddings, embedding_dim), device=device, dtype=dtype),
            requires_grad=True,
        )

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.trunc_normal_(self.embeddings, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings[token_ids]
