from abc import ABC, abstractmethod
from typing import Optional
import os
import torch
from copy import deepcopy
from flwr.common.typing import Parameters, FitRes
from flwr.server.client_proxy import ClientProxy
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.logger import logger
from floral.utils import get_ndarrays, set_ndarrays
from .fedavg_base import FedAvgBase


class FedAvgWithRouter(FedAvgBase, ABC):
    def __init__(self, *args,
                 global_model: torch.nn.Module,
                 save_path: str,
                 global_lr: float = 1.0,
                 eps: float = 1e-10,
                 model_filename: Optional[str] = "model.pt",
                 **kwargs):
        assert global_model is not None
        super().__init__(*args, **kwargs)
        self.global_model = global_model
        self.save_path = save_path
        self.global_lr = global_lr
        self.eps = eps
        self.model_path = None
        if model_filename is not None:
            self.model_path = os.path.join(self.save_path, model_filename)

    @torch.no_grad()
    def aggregate(self,
                  results: list[tuple[ClientProxy, FitRes]],
                  ) -> Parameters:
        # Sort results by cid, save prev params if needed
        fit_results = list(sorted([
            (int(client_proxy.cid), parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for client_proxy, fit_res in results
        ]))
        # Save prev params to apply global lr later
        if self.global_lr != 1.0:
            prev_dict = deepcopy(self.global_model.state_dict())

        # Apply weights
        router_weights = self.get_router_weights(self.global_model, fit_results)
        fit_results = self.apply_weights_(self.global_model, router_weights, fit_results, eps=self.eps)
        # DEBUG: print routes per client
        for cid, weight in router_weights.items():
            routes = weight.softmax(-1).mul(100).round().int().tolist()
            logger.debug(f"Client {cid}: routes = {routes}")
        # Aggregate
        all_parameters = [p for _, p, _ in fit_results]
        aggregated_weights = [sum(all_layers) for all_layers in zip(*all_parameters)]
        set_ndarrays(self.global_model, aggregated_weights)

        # XXX: Defuse
        if hasattr(self.global_model, "fuse_params") and self.global_model.fuse_params:
            for m in self.global_model.lora_modules.values():
                m.defuse()

        # Apply global lr
        if self.global_lr != 1.0:
            for p, p0 in zip(self.global_model.state_dict(), prev_dict.values()):
                if isinstance(p, torch.Tensor) and torch.is_floating_point(p) and \
                        isinstance(p0, torch.Tensor) and torch.is_floating_point(p0):
                    p.copy_(p0.sub(p0.sub(p), alpha=self.global_lr))

        # Save global model and return parameters
        if self.model_path is not None:
            torch.save(self.global_model.state_dict(), self.model_path)

        global_weights = deepcopy(get_ndarrays(self.global_model))
        global_parameters = ndarrays_to_parameters(global_weights)

        return global_parameters

    @staticmethod
    @torch.no_grad()
    def get_router_weights(global_model: torch.nn.Module,
                            fit_results: list[tuple[int, ClientProxy, FitRes]],
                            ) -> dict[int, torch.Tensor]:
        # TODO(optimization): loading the full model just to get one small vector is too much
        router_weights = {}
        for result_idx in range(len(fit_results)):
            cid, parameters, _ = fit_results[result_idx]
            set_ndarrays(global_model, parameters)
            router_weights[cid] = global_model.router.weight.clone()
        return router_weights

    @staticmethod
    @abstractmethod
    @torch.no_grad()
    def apply_weights_(global_model: torch.nn.Module,
                       router_weights: dict[int, torch.Tensor],
                       fit_results: list[tuple[int, ClientProxy, FitRes]],
                       eps: float = 1e-10,
                       ) -> list[tuple[int, ClientProxy, FitRes]]:
        raise NotImplementedError("Weighting mechanism for FedAvgWithRouter is not implemented.")
