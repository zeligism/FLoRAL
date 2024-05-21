import os
from copy import deepcopy
import torch
from flwr.common.typing import NDArrays
from .flwr_client_pvt import FlowerClientWithPrivateModules


class MomentumClient(FlowerClientWithPrivateModules):
    def __init__(self, *args, momentum: float = 0.9, grad_momentum: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.momentum = momentum
        self.grad_momentum = grad_momentum
        self.optim_path = os.path.join(
            self.save_path, self.private_dir, f"{self.client_id}_optim.pt")
        if os.path.exists(self.optim_path):
            optim_state = torch.load(self.optim_path)
            self.exp_avg = optim_state["exp_avg"]
        else:
            self.exp_avg = {}
        self.state_dict0 = {}

    def get_parameters(self, *args, **kwargs) -> NDArrays:
        # calculate momentum
        with torch.no_grad():
            state_dict = self.model.state_dict()
            for k in state_dict.keys():
                if state_dict[k] is None or k not in self.state_dict0 or self.state_dict0[k] is None:
                    continue
                if self.grad_momentum:
                    delta = self.state_dict0[k].sub(state_dict[k])
                    if k not in self.exp_avg:
                        self.exp_avg[k] = delta
                    self.exp_avg[k].mul_(self.momentum).add_(delta.mul(1 - self.momentum))
                    state_dict[k].copy_(self.state_dict0[k].sub(self.exp_avg[k]))
                else:
                    if k not in self.exp_avg:
                        self.exp_avg[k] = state_dict[k].clone()
                    self.exp_avg[k] = self.exp_avg[k].mul(self.momentum).add(
                        state_dict[k].mul(1 - self.momentum)
                    )
                    state_dict[k].copy_(self.exp_avg[k])
        optim_state = {"exp_avg": self.exp_avg}
        torch.save(optim_state, self.optim_path)
        return super().get_parameters(*args, **kwargs)

    def set_parameters(self, *args, **kwargs) -> None:
        super().set_parameters(*args, **kwargs)
        self.state_dict0 = deepcopy(self.model.state_dict())


