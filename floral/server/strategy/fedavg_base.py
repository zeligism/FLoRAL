import os
import torch
import json
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union
from flwr.common import EvaluateRes, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.common.typing import Parameters, Scalar, FitRes, Metrics
from flwr.server.client_proxy import ClientProxy
from flwr.common.logger import logger
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate, aggregate_inplace
from floral.utils import get_ndarrays, set_ndarrays
from ..client_manager import ClientManagerBase

MODEL_FNAME = "model.pt"
OPTIMIZER_FNAME = "optimizer.pt"
HISTORY_PROGRESS_FNAME = "history.json"


class FedAvgBase(FedAvg):
    """Algorithmically identical to FedAvg, but with some extra convenient features."""
    def __init__(self, *args,
                 global_model: torch.nn.Module,
                 global_optimizer: torch.optim.Optimizer,
                 save_path: str,
                 continue_training: bool = False,
                 **kwargs):
        assert global_model is not None
        super().__init__(*args, **kwargs)
        self.global_model = global_model
        self.global_optimizer = global_optimizer
        self.save_path = save_path
        self.model_path = os.path.join(self.save_path, MODEL_FNAME)
        self.optimizer_path = os.path.join(self.save_path, OPTIMIZER_FNAME)
        self.history_progress_path = os.path.join(self.save_path, HISTORY_PROGRESS_FNAME)
        if continue_training:
            self._load_states()
        else:
            self.save_history({})
            self.last_round = 0

    def get_history(self, flwr_format=False):
        with open(self.history_progress_path) as f:
            history = json.load(f)
        # XXX: currently used for backward compatibility, deprecate later
        if flwr_format:
            for key in history:
                metrics_as_list = sorted([(int(round), vals) for round, vals in history[key].items()])
                history[key] = metrics_as_list

        return history

    def save_history(self, history):
        with open(self.history_progress_path, 'w') as f:
            json.dump(history, f, indent=2)

    def _load_states(self):
        device = next(self.global_model.parameters())[0].device
        if os.path.exists(self.model_path):
            logger.info(f"Loading model from: {self.model_path}")
            model_state_dict = torch.load(self.model_path, map_location=device)
            self.global_model.load_state_dict(model_state_dict)
        else:
            logger.warning(f"Couldn't find a pretrained model at {self.model_path}")
            logger.warning("Using current model")

        if os.path.exists(self.optimizer_path):
            logger.info(f"Loading optimizer from: {self.optimizer_path}")
            optimizer_state_dict = torch.load(self.optimizer_path, map_location=device)
            self.global_optimizer.load_state_dict(optimizer_state_dict)
        else:
            logger.warning(f"Couldn't find an optimizer state at {self.optimizer_path}")
            logger.warning("Using current optimizer")

        if os.path.exists(self.history_progress_path):
            with open(self.history_progress_path) as f:
                history = json.load(f)
            all_rounds = [int(round) for metric_history in history.values() for round in metric_history.keys()]
            self.last_round = max([0] + all_rounds)
        else:
            logger.warning(f"Couldn't find a history in progress at {self.history_progress_path}")
            logger.warning("Starting a new history")
            self.save_history({})
            self.last_round = 0

        logger.info(f"Continue training from round {self.last_round + 1}")

    def save_history_progress(
            self,
            server_round: int,
            loss_aggregated: Optional[float] = None,
            metrics_aggregated: Optional[dict[str, Scalar]] = None,
            metrics_aggregated_fit: Optional[dict[str, Scalar]] = None,
            ) -> None:
        # Construct history point
        history_point = {}
        if loss_aggregated is not None:
            history_point["losses_distributed"] = loss_aggregated
        if metrics_aggregated is not None:
            for key, value in metrics_aggregated.items():
                history_point[key + "_distributed"] = value
        if metrics_aggregated_fit is not None:
            for key, value in metrics_aggregated_fit.items():
                history_point[key + "_distributed_fit"] = value

        # Load history, append history point, and save
        overall_server_round = self.last_round + server_round
        history = self.get_history()
        for key, value in history_point.items():
            if key not in history:
                history[key] = {}
            history[key][overall_server_round] = value
        self.save_history(history)

    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        return ndarrays_to_parameters(get_ndarrays(self.global_model))

    def configure_fit(self, server_round, parameters, client_manager: ClientManagerBase):
        client_manager.set_phase("fit")
        return super().configure_fit(server_round, parameters, client_manager)

    def configure_evaluate(self, server_round, parameters, client_manager: ClientManagerBase):
        client_manager.set_phase("evaluate")
        return super().configure_evaluate(server_round, parameters, client_manager)

    @torch.no_grad()
    def aggregate(self, results: list[tuple[ClientProxy, FitRes]]) -> Parameters:
        # Aggregate as in FedAvg
        if self.inplace:
            aggregated_ndarrays = aggregate_inplace(results)
        else:
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in results
            ]
            aggregated_ndarrays = aggregate(weights_results)

        # Apply the aggregation using the global optimizer
        prev_params = deepcopy(list(self.global_model.parameters()))
        set_ndarrays(self.global_model, aggregated_ndarrays)
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

    # ---------- The following methods are completely identical to FedAvg, except where indicated ---------- #
    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        parameters_aggregated = self.aggregate(results)  # <---------- CUSTOM AGGREGATOR

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            logger.warning("No fit_metrics_aggregation_fn provided")
        # Save metrics to history-in-progress file. <---------- SAVE METRICS IN_PROGRESS
        self.save_history_progress(server_round, metrics_aggregated_fit=metrics_aggregated)

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[Tuple[ClientProxy, EvaluateRes] | BaseException]
            ) -> Tuple[float | None, Dict[str, bool | bytes | float | int | str]]:
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(server_round, results, failures)
        # Save metrics to history-in-progress file. <---------- SAVE METRICS IN_PROGRESS
        self.save_history_progress(server_round, loss_aggregated=loss_aggregated,
                                   metrics_aggregated=metrics_aggregated)

        return loss_aggregated, metrics_aggregated

