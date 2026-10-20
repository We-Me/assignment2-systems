from .naive_ddp import NaiveDDP
from .flat_ddp import FlatDDP
from .overlap_ddp import OverlapDDP


DDP = OverlapDDP

__all__ = [
    "NaiveDDP",
    "FlatDDP",
    "OverlapDDP",
    "DDP",
]
