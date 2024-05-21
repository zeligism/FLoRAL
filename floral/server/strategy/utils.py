from collections import defaultdict
from omegaconf import OmegaConf
from hydra.utils import instantiate
from flwr.common.typing import Metrics
from flwr.common import ndarrays_to_parameters
from floral.utils import get_ndarrays


# Define metric aggregation function
def get_metrics_aggregation_fn():
    """This function is used to aggregate both fit and eval metrics."""
    def weighted_average(metrics_per_client: list[tuple[int, Metrics]]) -> Metrics:
        """Compute per-dataset average accuracy and loss."""
        n = sum(n_i for n_i, _ in metrics_per_client)
        aggregated_metrics = defaultdict(float)
        for n_i, metrics in metrics_per_client:
            w_i = n_i / n
            for k in metrics.keys():
                aggregated_metrics[k] += w_i * metrics[k]

        return aggregated_metrics

    return weighted_average


def get_on_fit_config_fn(cfg):
    """Return a config (a dict) to be sent to clients during fit()."""
    if "finetune_after" in cfg and cfg.finetune_after is not None:
        finetune_round = float(cfg.finetune_after)
        if finetune_round < 1:
            finetune_round = max(1, finetune_round * cfg.num_rounds)
        else:
            finetune_round = finetune_round
    else:
        finetune_round = 10**10

    def on_fit_config_fn(server_round: int):
        fit_config = {}
        fit_config["round"] = server_round
        fit_config["finetune_round"] = finetune_round
        return fit_config

    return on_fit_config_fn


def get_on_evaluate_config_fn(cfg):
    def on_evaluate_config_fn(server_round: int):
        eval_config = {}
        eval_config["round"] = server_round
        return eval_config

    return on_evaluate_config_fn


def get_evaluate_fn(cfg):
    # TODO(main): maybe we will need a central evaluate_fn
    # def evaluate_fn(server_round, parameters, config):
    #     start_time = now()
    #     loss, _, metrics = eval_client.evaluate(parameters, config)
    #     return loss, metrics

    return None


def get_strategy(cfg, global_model, save_path):
    strategy_opts = {
        "initial_parameters": ndarrays_to_parameters(get_ndarrays(global_model)),
        "evaluate_fn": get_evaluate_fn(cfg),
        "on_fit_config_fn": get_on_fit_config_fn(cfg),
        "on_evaluate_config_fn": get_on_evaluate_config_fn(cfg),
    }
    if OmegaConf.is_missing(cfg.strategy, "global_model"):
        strategy_opts["global_model"] = global_model
    if OmegaConf.is_missing(cfg.strategy, "save_path"):
        strategy_opts["save_path"] = save_path

    return instantiate(cfg.strategy, **strategy_opts)
