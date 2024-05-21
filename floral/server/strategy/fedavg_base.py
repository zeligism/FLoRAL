from typing import Dict, List, Optional, Tuple, Union
from flwr.common import EvaluateRes, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.common.typing import Parameters, Scalar, FitRes
from flwr.server.client_proxy import ClientProxy
from flwr.common.logger import logger
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate, aggregate_inplace
from ..client_manager import ClientManagerBase


class FedAvgBase(FedAvg):
    """Almost identical to FedAvg, but with some customization"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def configure_fit(self, server_round, parameters, client_manager: ClientManagerBase):
        client_manager.set_phase("fit")
        return super().configure_fit(server_round, parameters, client_manager)

    def configure_evaluate(self, server_round, parameters, client_manager: ClientManagerBase):
        client_manager.set_phase("evaluate")
        return super().configure_evaluate(server_round, parameters, client_manager)
        

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

        return parameters_aggregated, metrics_aggregated

    # def aggregate_evaluate(
    #         self,
    #         server_round: int,
    #         results: List[Tuple[ClientProxy, EvaluateRes]],
    #         failures: List[Tuple[ClientProxy, EvaluateRes] | BaseException]
    #         ) -> Tuple[float | None, Dict[str, bool | bytes | float | int | str]]:
    #     loss_aggregated, metrics_aggregated = super().aggregate_evaluate(server_round, results, failures)
    #     return loss_aggregated, metrics_aggregated

    # This is straight up copy-and-paste from FedAvg
    def aggregate(self, results: list[tuple[ClientProxy, FitRes]]) -> Parameters:
        if self.inplace:
            # Does in-place weighted average of results
            aggregated_ndarrays = aggregate_inplace(results)
        else:
            # Convert results
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in results
            ]
            aggregated_ndarrays = aggregate(weights_results)

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        return parameters_aggregated
