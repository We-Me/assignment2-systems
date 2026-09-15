import math
from typing import Callable, Optional

import torch
from torch.optim.optimizer import ParamsT


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.99),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p] = {
                    "m": torch.zeros_like(p, dtype=p.dtype, device=p.device),
                    "v": torch.zeros_like(p, dtype=p.dtype, device=p.device),
                    "t": 1
                }

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                t = state.get("t")
                beta1_hat = math.pow(betas[0], t)
                beta2_hat = math.pow(betas[1], t)
                lr_hat = lr * math.sqrt(1 - beta2_hat) / (1 - beta1_hat)
                state["t"] = t + 1

                p.data -= lr * weight_decay * p.data

                m = state.get("m")
                m_hat = betas[0] * m + (1 - betas[0]) * grad
                state["m"] = m_hat
                v = state.get("v")
                v_hat = betas[1] * v + (1 - betas[1]) * torch.pow(grad, 2)
                state["v"] = v_hat

                p.data -= lr_hat * torch.divide(m_hat, (torch.sqrt(v_hat) + eps))

        return loss
