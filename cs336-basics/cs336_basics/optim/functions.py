from collections.abc import Iterable
import math

import torch


def lr_cosine_schedule(
    t: int,
    alpha_min: float,
    alpha_max: float,
    warmup_iters: int,
    cosine_iters: int,
):
    if t < warmup_iters:
        return alpha_max * t / warmup_iters
    elif t > cosine_iters:
        return alpha_min
    else:
        return (
                alpha_min +
                (alpha_max - alpha_min) * 0.5 * (
                    1 +
                    math.cos(math.pi * (t - warmup_iters) / (cosine_iters - warmup_iters))
                )
            )


@torch.no_grad()
def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6
):
    has_grad: list[torch.nn.Parameter] = [p for p in parameters if p.grad is not None]
    if not has_grad:
        return

    sum_sq = has_grad[0].grad.float().square().sum()
    for p in has_grad[1:]:
        sum_sq += p.grad.float().square().sum()

    global_norm_g = math.sqrt(sum_sq.item())
    if global_norm_g < max_l2_norm:
        return

    scaled = max_l2_norm / (global_norm_g + eps)
    for p in has_grad:
        p.grad.mul_(scaled)
