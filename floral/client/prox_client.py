import os
import torch
from typing import Optional
from flwr.common.typing import NDArrays, Scalar
from .flwr_client_pvt import FlowerClientWithPrivateModules
from floral.utils import dict_to_ndarrays, ndarrays_to_dict


class ProxClient(FlowerClientWithPrivateModules):
    def __init__(self, *args,
                 fedprox: bool = False,
                 ditto: bool = False,
                 prox_lambda=1.0,
                 prox_ord=2,
                 **kwargs) -> None:
        # kwargs["full_comm"] = True  # need full communication of all parameters
        super().__init__(*args, **kwargs)
        self.fedprox = fedprox
        self.ditto = ditto
        self.prox_lambda = prox_lambda
        self.prox_ord = prox_ord
        self.personal_model_path = os.path.join(
            self.save_path, self.private_dir, f"personal_model{self.client_id}.pt")

    @staticmethod
    def get_prox_regularizer(prox_state_dict: dict[int, torch.Tensor], ord = 'fro'):
        def prox_regularizer(model: torch.nn.Module):
            prox = []
            for name, param in model.named_parameters():
                if name in prox_state_dict and prox_state_dict[name] is not None:
                    dist = param.sub(prox_state_dict[name].detach())
                    if ord == 'nuc':
                        prox.append(dist.view(dist.size(0), -1).norm(p='nuc'))
                    elif ord == 'fro' or float(ord) == 2.0:
                        prox.append(dist.pow(2).sum().mul(0.5))
                    elif float(ord) == 1.0:
                        prox.append(dist.abs().sum())
                    else:
                        prox.append(dist.pow(float(ord)).sum())
            return torch.stack(prox).sum()

        return prox_regularizer

    def set_prox_regularizer(self, prox_parameters: Optional[NDArrays]) -> None:
        if prox_parameters is not None:
            prox_state_dict = ndarrays_to_dict(self.model.state_dict().keys(), prox_parameters)
            prox_regularizer = self.get_prox_regularizer(prox_state_dict, ord=self.prox_ord)
            self.trainer.regularizer.add_regularizer("prox", self.prox_lambda, prox_regularizer)
        else:
            self.trainer.regularizer.remove_regularizer("prox")

    def fit(self, global_parameters: NDArrays, config: dict[str, Scalar]):
        # Train for global model (with prox if fedprox)
        self.set_prox_regularizer(global_parameters if self.fedprox else None)
        fitted_global_parameters, num_data, global_metrics = super().fit(global_parameters, config)

        # Fine-tune local model (with prox if ditto) --- #
        if os.path.exists(self.personal_model_path):
            personal_parameters = dict_to_ndarrays(torch.load(self.personal_model_path))
            self.set_prox_regularizer(global_parameters if self.ditto else None)
            fitted_personal_parameters, _, personal_metrics = super().fit(personal_parameters, config)
        else:
            fitted_personal_parameters, personal_metrics = fitted_global_parameters, global_metrics
        # Save personal model
        personal_state_dict = ndarrays_to_dict(self.model.state_dict().keys(), fitted_personal_parameters)
        torch.save(personal_state_dict, self.personal_model_path)

        return fitted_global_parameters, num_data, personal_metrics


class FedProxClient(ProxClient):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["ditto"] = False
        kwargs["fedprox"] = True
        super().__init__(*args, **kwargs)


class DittoClient(ProxClient):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["ditto"] = True
        kwargs["fedprox"] = False
        super().__init__(*args, **kwargs)

