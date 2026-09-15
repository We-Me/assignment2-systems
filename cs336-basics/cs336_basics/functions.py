import os
from typing import IO, BinaryIO

import torch
import numpy as np
import numpy.typing as npt


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_values, _ = torch.max(input=x, dim=dim, keepdim=True)
    x = x - max_values
    x = torch.exp(input=x)
    x_sum = x.sum(dim=dim, keepdim=True)
    return x / x_sum


def softmax_with_temperature(x: torch.Tensor, dim: int, t: float, eps: float = 1e-8) -> torch.Tensor:
    max_values, _ = torch.max(input=x, dim=dim, keepdim=True)
    x = x - max_values
    x = x / (t + eps)
    x = torch.exp(input=x)
    x_sum = x.sum(dim=dim, keepdim=True)
    return x / x_sum


def cross_entropy_loss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    max_values, _ = torch.max(input=inputs, dim=-1, keepdim=True)
    logits = inputs - max_values
    loss = logits.exp().sum(dim=-1, keepdim=True).log() - logits.gather(dim=-1, index=targets.unsqueeze(-1))
    return loss.mean(dim=0)


def silu(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x) * x


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str
):
    device = torch.device(device)

    bound = len(dataset) - (context_length + 1) + 1
    gen = np.random.default_rng()
    selected = gen.integers(low=0, high=bound, size=batch_size)

    inputs = []
    targets = []
    for s in selected:
        inputs.append(torch.tensor(dataset[s: s + context_length], device=device, dtype=torch.long))
        targets.append(torch.tensor(dataset[s + 1: s + context_length + 1], device=device, dtype=torch.long))
    inputs = torch.stack(inputs, dim=0)
    targets = torch.stack(targets, dim=0)

    return inputs, targets


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    state_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration
    }

    torch.save(state_dict, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    state_dict = torch.load(src)
    model.load_state_dict(state_dict=state_dict["model"])
    optimizer.load_state_dict(state_dict=state_dict["optimizer"])
    return state_dict["iteration"]
