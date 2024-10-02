import os
from copy import deepcopy
import torch
from flwr.common.typing import NDArrays
from .flwr_client_pvt import FlowerClientWithPrivateModules


class AdamClient(FlowerClientWithPrivateModules):
    exp_avg: dict[str, torch.Tensor]
    exp_avg_sq: dict[str, torch.Tensor]
    incoming_state_dict: dict[str, torch.Tensor]

    def __init__(self, *args,
                 betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.betas = betas
        self.eps = eps
        self.optim_path = os.path.join(
            self.save_path, self.private_dir, f"{self.client_id}_optim.pt")
        if os.path.exists(self.optim_path):
            optim_dict = torch.load(self.optim_path)
            self.exp_avg = optim_dict["exp_avg"]
            self.exp_avg_sq = optim_dict["exp_avg_sq"]
        else:
            self.exp_avg = {"_iters": 1}
            self.exp_avg_sq = {"_iters": 1}
        self.incoming_state_dict = {}

    def set_parameters(self, *args, **kwargs) -> None:
        super().set_parameters(*args, **kwargs)
        self.incoming_state_dict = deepcopy(self.model.state_dict())

    def get_parameters(self, *args, **kwargs) -> NDArrays:
        # calculate momentum
        with torch.no_grad():
            # bias correction
            bias_corr = 1 - self.betas[0] ** self.exp_avg["_iters"]
            bias_corr_sq = 1 - self.betas[1] ** self.exp_avg_sq["_iters"]
            bias_corr_factor = bias_corr_sq ** 0.5 / bias_corr
            self.exp_avg["_iters"] += 1
            self.exp_avg_sq["_iters"] += 1
            outgoing_state_dict = self.model.state_dict()
            for k, p in outgoing_state_dict.items():
                p_prev = self.incoming_state_dict[k]
                if p is None or p_prev is None:
                    continue
                if not isinstance(p, torch.Tensor) and not torch.is_floating_point(p):
                    continue
                # pseudo_grad
                pseudo_grad = p_prev.sub(p)
                pseudo_grad_sq = pseudo_grad.square()
                # exp_avg
                if k not in self.exp_avg:
                    self.exp_avg[k] = pseudo_grad
                else:
                    self.exp_avg[k] = self.exp_avg[k].mul(self.betas[0]).add(
                        pseudo_grad.mul(1 - self.betas[0]))
                # exp_avg_sq
                if k not in self.exp_avg_sq:
                    self.exp_avg_sq[k] = pseudo_grad_sq
                else:
                    self.exp_avg_sq[k] = self.exp_avg_sq[k].mul(self.betas[1]).add(
                        pseudo_grad_sq.mul(1 - self.betas[1]))
                # Adam step
                adam_pseudo_grad = self.exp_avg[k].div(
                    self.exp_avg_sq[k].sqrt().add(self.eps)
                ).mul(bias_corr_factor)
                p.copy_(p_prev.sub(adam_pseudo_grad))
        # save client optim state
        optim_state = {"exp_avg": self.exp_avg, "exp_avg_sq": self.exp_avg_sq}
        torch.save(optim_state, self.optim_path)
        return super().get_parameters(*args, **kwargs)
