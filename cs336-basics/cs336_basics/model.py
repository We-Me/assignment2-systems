from torch import nn

from cs336_basics.modules import Embedding as _Embedding
from cs336_basics.modules import Linear as _Linear
from cs336_basics.modules import RMSNorm


class Linear(_Linear):
    @property
    def weight(self) -> nn.Parameter:
        return self.weights

    @weight.setter
    def weight(self, value: nn.Parameter) -> None:
        self.weights = value


class Embedding(_Embedding):
    @property
    def weight(self) -> nn.Parameter:
        return self.embeddings

    @weight.setter
    def weight(self, value: nn.Parameter) -> None:
        self.embeddings = value


__all__ = [
    "Embedding",
    "Linear",
    "RMSNorm",
]
