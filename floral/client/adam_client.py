import os
from copy import deepcopy
import torch
from flwr.common.typing import NDArrays
from .flwr_client import FlowerClient


class AdamClient(FlowerClient):
    def __init__(self, *args,
                 betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, alpha: float = 0.01,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.betas = betas
        self.eps = eps
        self.alpha = alpha
        self.optim_path = os.path.join(self.private_dir, f"{self.client_id}_optim.pt")
        if os.path.exists(self.optim_path):
            optim_dict = torch.load(self.optim_path)
            self.exp_avg = optim_dict["exp_avg"]
            self.exp_avg_sq = optim_dict["exp_avg_sq"]
        else:
            self.exp_avg = {"_iters": 1}
            self.exp_avg_sq = {"_iters": 1}
        self.state_dict0 = {}

    def get_parameters(self, *args, **kwargs) -> NDArrays:
        # calculate momentum
        state_dict = self.model.state_dict()
        with torch.no_grad():
            # bias correction
            bias_corr = 1 - self.betas[0] ** self.exp_avg["_iters"]
            bias_corr_sq = 1 - self.betas[1] ** self.exp_avg_sq["_iters"]
            bias_corr_factor = bias_corr_sq ** 0.5 / bias_corr
            self.exp_avg["_iters"] += 1
            self.exp_avg_sq["_iters"] += 1
            for k in state_dict.keys():
                if state_dict[k] is None or k not in self.state_dict0 or self.state_dict0[k] is None:
                    continue
                # pseudo_grad
                pseudo_grad = self.state_dict0[k].sub(state_dict[k])
                pseudo_grad_sq = pseudo_grad.pow(2)
                # exp_avg
                if k not in self.exp_avg:
                    self.exp_avg[k] = pseudo_grad
                self.exp_avg[k] = self.exp_avg[k].mul(self.betas[0]).add(
                    pseudo_grad.mul(1 - self.betas[0])
                )
                # exp_avg_sq
                if k not in self.exp_avg_sq:
                    self.exp_avg_sq[k] = pseudo_grad_sq
                self.exp_avg_sq[k] = self.exp_avg_sq[k].mul(self.betas[1]).add(
                    pseudo_grad_sq.mul(1 - self.betas[1])
                )
                # Adam step
                adam_pseudo_grad = self.exp_avg[k].div(self.exp_avg_sq[k].sqrt().add(self.eps)).mul(bias_corr_factor)
                state_dict[k].copy_(self.state_dict0[k].sub(adam_pseudo_grad, alpha=self.alpha))
        # save file
        optim_state = {"exp_avg": self.exp_avg, "exp_avg_sq": self.exp_avg_sq}
        torch.save(optim_state, self.optim_path)
        # get parameters
        return super().get_parameters(*args, **kwargs)

    def set_parameters(self, *args, **kwargs) -> None:
        super().set_parameters(*args, **kwargs)
        self.state_dict0 = deepcopy(self.model.state_dict())
