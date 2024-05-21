
from flwr.common.typing import NDArrays, Scalar
from floral.utils import split_state_dict
from .flwr_client_pvt import FlowerClientWithPrivateModules


class FlowerClientWithPersonalizedModules(FlowerClientWithPrivateModules):
    def __init__(self, *args, split_update_budget=False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.split_update_budget = split_update_budget

    def personalization_mode(self, on: bool = True):
        public_dict, private_dict = split_state_dict(
            self.model.state_dict(), self._private_modules)
        for p in public_dict.values():
            if hasattr(p, "requires_grad_"):
                p.requires_grad_(not on)
        for p in private_dict.values():
            if hasattr(p, "requires_grad_"):
                p.requires_grad_(on)

    def fit(self, parameters: NDArrays, config: dict[str, Scalar]) -> tuple[NDArrays, int, dict]:
        if self.split_update_budget:
            # Split the update budget equally among the two stages
            local_epochs = self.local_epochs
            self.local_epochs = local_epochs / 2
        self.personalization_mode(True)
        _ = super().fit(parameters, config)
        self.personalization_mode(False)
        results = super().fit(parameters, config)
        if self.split_update_budget:
            self.local_epochs = local_epochs  # reset
        return results


class FlowerClientWithFinetuning(FlowerClientWithPrivateModules):
    def fit(self, parameters: NDArrays, config: dict[str, Scalar]) -> tuple[NDArrays, int, dict]:
        if config["round"] > config["finetune_round"]:
            # Training becomes fully local after `finetune_round` rounds
            self.set_private_modules([""])
        results = super().fit(parameters, config)
        return results
