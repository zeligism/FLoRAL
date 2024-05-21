import torch
from copy import deepcopy
from flwr.common.typing import FitRes
from flwr.server.client_proxy import ClientProxy
from floral.utils import get_ndarrays, set_ndarrays
from .fedavg_router import FedAvgWithRouter


class EnsembleAvg(FedAvgWithRouter):
    @staticmethod
    def apply_weights_(global_model: torch.nn.Module,
                       router_weights: dict[int, torch.Tensor],
                       fit_results: list[tuple[int, ClientProxy, FitRes]],
                       eps: float = 1e-10,
                       ) -> list[tuple[int, ClientProxy, FitRes]]:
        # --- Calculate aggregation weights, including cluster-wise weights
        client_loads = {cid: num_examples for cid, _, num_examples in fit_results}
        client_probs = {cid: load / (sum(client_loads.values()) + eps) for cid, load in client_loads.items()}
        cluster_probs = {cid: w.softmax(dim=-1) for cid, w in router_weights.items()}
        client_cluster_loads = {cid: client_loads[cid] * cluster_probs[cid]
                                for cid in client_loads.keys() if cid in cluster_probs}
        client_cluster_probs = {cid: cluster_loads / (sum(client_cluster_loads.values()) + eps)
                                for cid, cluster_loads in client_cluster_loads.items()}

        # --- Apply aggregation weights
        for result_idx in range(len(fit_results)):
            cid, parameters, num_examples = fit_results[result_idx]
            set_ndarrays(global_model, parameters)
            # Shared
            for cluster_id, model in enumerate(global_model.models):
                for p in model.state_dict().values():
                    if isinstance(p, torch.Tensor) and torch.is_floating_point(p):
                        if cid in client_cluster_probs:
                            w_kc = client_cluster_probs[cid][cluster_id]
                        else:
                            w_kc = client_probs[cid]
                        p.mul_(w_kc)
            parameters_reweighted_lora = deepcopy(get_ndarrays(global_model))
            fit_results[result_idx] = cid, parameters_reweighted_lora, num_examples
        
        return fit_results
