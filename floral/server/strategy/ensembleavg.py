import torch
from copy import deepcopy
from flwr.common.typing import Any, FitRes
from flwr.server.client_proxy import ClientProxy
from floral.utils import get_ndarrays, set_ndarrays
from .fedavg_router import FedAvgWithRouter, get_client_cluster_probs


class EnsembleAvg(FedAvgWithRouter):
    @staticmethod
    def apply_weights_(global_model: torch.nn.Module,
                       probs: dict[int, Any],
                       fit_results: list[tuple[int, ClientProxy, FitRes]],
                       ) -> list[tuple[int, ClientProxy, FitRes]]:        
        for result_idx in range(len(fit_results)):
            cid, parameters, num_examples = fit_results[result_idx]
            set_ndarrays(global_model, parameters)
            # Shared
            for cluster_id, model in enumerate(global_model.models):
                for p in model.state_dict().values():
                    if isinstance(p, torch.Tensor) and torch.is_floating_point(p):
                        if cid in probs["client_given_cluster"]:
                            w_kc = probs["client_given_cluster"][cid][cluster_id]
                        else:
                            w_kc = probs["client"][cid]
                        p.mul_(w_kc)
            parameters_reweighted_lora = deepcopy(get_ndarrays(global_model))
            fit_results[result_idx] = cid, parameters_reweighted_lora, num_examples
        
        return fit_results
