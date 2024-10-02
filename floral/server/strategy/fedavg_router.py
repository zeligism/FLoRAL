from abc import ABC, abstractmethod
from typing import Optional
import os
import torch
import json
from copy import deepcopy
from flwr.common.typing import Any, NDArrays, Parameters, FitRes, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.logger import logger
from floral.utils import get_ndarrays, set_ndarrays
from .fedavg_base import FedAvgBase


class FedAvgWithRouter(FedAvgBase, ABC):
    @torch.no_grad()
    def aggregate(self,
                  results: list[tuple[ClientProxy, FitRes]],
                  ) -> Parameters:
        # Sort results by cid, save prev params if needed
        fit_results = list(sorted([
            (int(client_proxy.cid), parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for client_proxy, fit_res in results
        ]))
        prev_params = deepcopy(list(self.global_model.parameters()))

        # Apply aggregation weights to models
        router_weights = self.get_router_weights(self.global_model, fit_results)
        probs = get_client_cluster_probs(router_weights=router_weights, fit_results=fit_results)
        fit_results = self.apply_weights_(self.global_model, probs, fit_results)
        # DEBUG: print routes per client
        for cid, weight in router_weights.items():
            routes = weight.softmax(-1).mul(100).round().int().tolist()
            logger.debug(f"Client {cid}: routes = {routes}")
        # Sum
        clients_parameters = [p for _, p, _ in fit_results]
        aggregated_weights = [sum(clients_layers) for clients_layers in zip(*clients_parameters)]
        set_ndarrays(self.global_model, aggregated_weights)

        # TODO: lora centering
        # if hasattr(self.global_model, "center_loras_at_base_"):
        #     self.global_model.center_loras_at_base_(probs["cluster"])

        # # XXX: Defuse
        # if hasattr(self.global_model, "fuse_params") and self.global_model.fuse_params:
        #     for m in self.global_model.lora_modules.values():
        #         m.defuse()

        # Set grad and step
        self.global_optimizer.zero_grad()
        for p, p0 in zip(self.global_model.parameters(), prev_params):
            p.grad = p0.sub(p)
            p.copy_(p0)
        self.global_optimizer.step()

        # Save and return ndarrays as parameters
        torch.save(self.global_model.state_dict(), self.model_path)
        torch.save(self.global_optimizer.state_dict(), self.optimizer_path)
        global_weights = deepcopy(get_ndarrays(self.global_model))
        global_parameters = ndarrays_to_parameters(global_weights)

        return global_parameters

    @staticmethod
    @torch.no_grad()
    def get_router_weights(global_model: torch.nn.Module,
                            fit_results: list[tuple[int, ClientProxy, FitRes]],
                            ) -> dict[int, torch.Tensor]:
        # TODO(optimization): loading the full model just to get one small vector may be a bit too much
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
                       probs: dict[str, Any],
                       fit_results: list[tuple[int, NDArrays, int]],
                       ) -> list[tuple[int, NDArrays, int]]:
        raise NotImplementedError("Weighting mechanism for FedAvgWithRouter is not implemented.")


def get_client_cluster_probs(router_weights: dict[int, torch.Tensor],
                             fit_results: list[tuple[int, ClientProxy, FitRes]],
                             eps: float = 1e-10,  # XXX: is this needed?
                             ) -> dict[str, Any]:
    
        # --- Calculate aggregation weights, including cluster-wise weights
        client_loads = {cid: num_examples for cid, _, num_examples in fit_results}
        client_loads_sum = sum(client_loads.values())
        # K -> p(k)
        client_probs = {cid: load / (client_loads_sum + eps) for cid, load in client_loads.items()}
        # K x C -> p(c|k)
        cluster_given_client_probs = {cid: w.softmax(dim=-1) for cid, w in router_weights.items()}
        # K x C -> p(k,c)
        client_cluster_probs = {cid: client_probs[cid] * cluster_given_client_probs[cid]
                                for cid in client_probs.keys() if cid in cluster_given_client_probs}
        # C -> p(c)
        cluster_probs = sum(client_cluster_probs.values())
        # K x C -> p(k|c)
        client_given_cluster_probs = {cid: client_cluster_probs[cid] / cluster_probs
                                      for cid in client_probs.keys() if cid in cluster_given_client_probs}

        return {
            "client": client_probs,  # p(k): dict[int, int]
            "cluster": cluster_probs,  # p(c): Tensor[C]
            "client_given_cluster": client_given_cluster_probs,  # p(k|c): dict[int, Tensor[C]]
            "cluster_given_client": cluster_given_client_probs,  # p(c|k): dict[int, Tensor[C]]
        }

