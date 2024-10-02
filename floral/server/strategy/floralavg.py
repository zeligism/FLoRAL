import torch
from copy import deepcopy
from flwr.common.typing import Any, FitRes
from flwr.server.client_proxy import ClientProxy
from floral.floral import Floral
from floral.utils import get_ndarrays, set_ndarrays
from .fedavg_router import FedAvgWithRouter, get_client_cluster_probs


class FloralAvg(FedAvgWithRouter):
    @staticmethod
    def apply_weights_(global_model: Floral,
                       probs: dict[str, Any],
                       fit_results: list[tuple[int, ClientProxy, FitRes]],
                       ) -> list[tuple[int, ClientProxy, FitRes]]:
        for result_idx in range(len(fit_results)):
            cid, parameters, num_examples = fit_results[result_idx]
            set_ndarrays(global_model, parameters)
            # Shared
            for name, p in global_model.base_model.state_dict().items():
                if isinstance(p, torch.Tensor) and torch.is_floating_point(p):
                    # e.g. num_batches_tracked are a part of the state dict
                    p.mul_(probs["client"][cid])
            # Clustered
            for m in global_model.lora_modules.values():
                for cluster_id in range(global_model.num_clusters):
                    if cid in probs["client_given_cluster"]:
                        prob = probs["client_given_cluster"][cid][cluster_id]
                    else:
                        prob = probs["client"][cid]
                    if hasattr(m, 'weight_in'):  # XXX: this case is deprecated, remove later
                        m.weight_in[cluster_id].mul_(prob)
                        m.weight_out[cluster_id].mul_(prob)
                        if m.has_bias:
                            m.bias[cluster_id].mul_(prob)
                    else:
                        lora = m[cluster_id]
                        for p in lora.state_dict().values():
                            if isinstance(p, torch.Tensor) and torch.is_floating_point(p):
                                p.mul_(prob)
            # Router is client-specific, so ignore
            for name, p in global_model.router.state_dict().items():
                if isinstance(p, torch.Tensor) and torch.is_floating_point(p):
                    p.mul_(0.0)
            parameters_reweighted_lora = deepcopy(get_ndarrays(global_model))
            fit_results[result_idx] = cid, parameters_reweighted_lora, num_examples

        return fit_results
