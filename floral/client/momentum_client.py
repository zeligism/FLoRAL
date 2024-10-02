import os
from copy import deepcopy
import torch
from flwr.common.typing import NDArrays
from .flwr_client_pvt import FlowerClientWithPrivateModules


class MomentumClient(FlowerClientWithPrivateModules):
    exp_avg: dict[str, torch.Tensor]
    incoming_state_dict: dict[str, torch.Tensor]

    def __init__(self, *args, momentum: float = 0.9, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.momentum = momentum
        self.optim_path = os.path.join(
            self.save_path, self.private_dir, f"{self.client_id}_optim.pt")
        if os.path.exists(self.optim_path):
            optim_state = torch.load(self.optim_path)
            self.exp_avg = optim_state["exp_avg"]
        else:
            self.exp_avg = {}
        self.incoming_state_dict = {}

    def set_parameters(self, *args, **kwargs) -> None:
        super().set_parameters(*args, **kwargs)
        self.incoming_state_dict = deepcopy(self.model.state_dict())

    def get_parameters(self, *args, **kwargs) -> NDArrays:
        # calculate momentum
        with torch.no_grad():
            outgoing_state_dict = self.model.state_dict()
            for k, p in outgoing_state_dict.items():
                p_prev = self.incoming_state_dict[k]
                if p is None or p_prev is None:
                    continue
                if not isinstance(p, torch.Tensor) and not torch.is_floating_point(p):
                    continue
                pseudo_grad = p_prev.sub(p)
                if k not in self.exp_avg:
                    self.exp_avg[k] = pseudo_grad
                else:
                    self.exp_avg[k].mul_(self.momentum).add_(pseudo_grad.mul(1 - self.momentum))
                p.copy_(p_prev.sub(self.exp_avg[k]))
        # save client optim state
        optim_state = {"exp_avg": self.exp_avg}
        torch.save(optim_state, self.optim_path)
        return super().get_parameters(*args, **kwargs)


