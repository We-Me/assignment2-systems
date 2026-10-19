from collections.abc import Callable
from typing import Any

import torch
import torch.distributed as dist


class ShardedOptimizer(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        optimizer_cls: type[torch.optim.Optimizer],
        **kwargs: Any,
    ) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("ShardedOptimizer requires an initialized torch.distributed process group")

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.optimizer_cls = optimizer_cls
        self._all_params: list[torch.nn.Parameter] = []
        self._param_owners: dict[int, int] = {}
        self._seen_params: set[int] = set()
        self._rank_loads: list[int] = [0] * self.world_size

        # Optimizer.__init__ calls add_param_group, which fills self.param_groups with this rank's local parameters.
        super().__init__(params, defaults=kwargs)
        self.optimizer: torch.optim.Optimizer = optimizer_cls(self.param_groups, **kwargs)

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        params = param_group["params"]
        if isinstance(params, torch.Tensor):
            params = [params]
        else:
            params = list(params)

        local_params = []
        for param in params:
            param_id = id(param)
            if param_id in self._seen_params:
                continue

            self._seen_params.add(param_id)
            # Greedily assign each parameter to the least-loaded rank by byte size.
            cost = param.numel() * param.element_size()
            owner = min(range(self.world_size), key=lambda candidate: self._rank_loads[candidate])
            self._rank_loads[owner] += cost
            self._all_params.append(param)
            self._param_owners[param_id] = owner

            if owner == self.rank:
                local_params.append(param)

        if not local_params:
            return

        local_group = {key: value for key, value in param_group.items() if key != "params"}
        local_group["params"] = local_params
        super().add_param_group(local_group)

        if getattr(self, "optimizer", None) is not None:
            self.optimizer.add_param_group(dict(local_group))

    def step(self, closure: Callable | None = None, **kwargs: Any):
        loss = self.optimizer.step(closure=closure, **kwargs)
        for parameter in self._all_params:
            owner = self._param_owners[id(parameter)]
            dist.broadcast(parameter.data, src=owner)
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        for parameter in self._all_params:
            if parameter.grad is None:
                continue
            if set_to_none:
                parameter.grad = None
            else:
                parameter.grad.detach_()
                parameter.grad.zero_()

    def state_dict(self) -> dict[str, Any]:
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state_dict)
