import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
import shutil
import pickle
from datetime import datetime
from pathlib import Path
from logging import getLevelName
# import getpass import getuser

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import flwr as fl
from flwr.common.logger import logger
from flwr.server import Server, ServerConfig
from floral.dataset import get_data
from floral.client import get_client_fn, get_on_actor_init_fn
from floral.server.strategy import get_strategy, FedAvgBase
from floral.server.client_manager import get_client_manager, ClientManagerBase
from floral.training.utils import instantiate_model, get_param_groups
from floral.utils import (
    init_seed,
    setup_wandb,
    get_ray_init_args,
    optuna_objective,
)

DEBUG = False
HISTORY_FILE = "history.pkl"
MAIN_LOG = "main.log"

# TODO(main): test on unseen clients
# TODO(main): check ditto
# TODO(later): better docs


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    hydra_cfg = HydraConfig.get()
    save_path = Path(hydra_cfg.runtime.output_dir)
    # Short-hand notations for _global_ cfgs
    cfg.task = hydra_cfg.runtime.choices["task@_global_"]
    cfg.method = hydra_cfg.runtime.choices["method@_global_"]
    cfg.extras = hydra_cfg.runtime.choices["extras@_global_"]

    # Configure logging
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
    cfg.logdir = os.path.join("logs", cfg.experiment, cfg.task_dir, timestamp)
    os.makedirs(cfg.logdir, exist_ok=True)
    filename = os.path.join(cfg.logdir, MAIN_LOG)
    fl.common.logger.update_console_handler(level=getLevelName(cfg.loglevel), timestamps=True, colored=True)
    fl.common.logger.configure(identifier=cfg.identifier, filename=filename)
    setup_wandb(cfg)  # setup wandb if it exists
    if cfg.show_cfg:
        logger.info("HYDRA CONFIG\n" + OmegaConf.to_yaml(hydra_cfg))
        logger.info("APP CONFIG\n" + OmegaConf.to_yaml(cfg))

    # If in multirun mode, check if results already exists and return recorded loss
    if hydra_cfg.mode == RunMode.MULTIRUN and not cfg.overwrite_sweep:
        history_path = os.path.join(save_path, HISTORY_FILE)
        if os.path.exists(history_path):
            logger.info(f"This run's history already exists: {history_path}")
            try:
                with open(history_path, "rb") as handle:
                    data = pickle.load(handle)
                    value = optuna_objective(data["history"])
                    logger.info(f"Loss = {value}")
                    return value
            except (pickle.UnpicklingError, KeyError):
                logger.info(f"Could not load history file: {history_path}")
                logger.info("Terminating this run with loss = inf")
                return optuna_objective(None)

    if "optimalrouter" in cfg.method and cfg.task in ("emnist", "shakespeare", "stackoverflow"):
        # mainly for sweeps runs
        logger.info(f"Dataset for task '{cfg.task}' does not have a known optimal router.")
        return optuna_objective(None)

    # ---------- Experiment setting ---------- #
    if cfg.deterministic:
        init_seed(cfg.seed)

    # Get initial global model and federated dataset
    global_model = instantiate_model(cfg)
    global_optimizer = instantiate(cfg.global_optimizer, get_param_groups(cfg, global_model))
    client_data = get_data(cfg)

    # Get client_fn
    client_fn, clients_ids = get_client_fn(cfg, client_data, save_path)
    # clients_ids, unseen_clients_ids = split_clients(clients_ids, unseen_clients=cfg.unseen_clients, seed=cfg.seed)
    fit_clients = [i for i, cid in clients_ids.items() if "train" in client_data[cid]]
    evaluate_clients = [i for i, cid in clients_ids.items() if "test" in client_data[cid]]

    if DEBUG:
        from floral.utils import get_ndarrays
        from flwr.common.typing import FitIns, EvaluateIns
        from flwr.common import ndarrays_to_parameters
        client_fn(fit_clients[0]).fit(FitIns(ndarrays_to_parameters(get_ndarrays(global_model)), {"round": 1}))
        client_fn(evaluate_clients[0]).evaluate(EvaluateIns(ndarrays_to_parameters(get_ndarrays(global_model)), {}))

    # Start simulation
    strategy: FedAvgBase = get_strategy(cfg, global_model, global_optimizer, save_path)
    client_manager: ClientManagerBase = get_client_manager(cfg, fit_clients=fit_clients, evaluate_clients=evaluate_clients)
    server = Server(client_manager=client_manager, strategy=strategy)
    simulation_opts = {
        "client_fn": client_fn,
        "clients_ids": clients_ids,
        "num_clients": len(clients_ids),
        "client_resources": cfg.client_resources,
        "server": server,
        # "config": ServerConfig(num_rounds=num_rounds, round_timeout=cfg.round_timeout),
        "ray_init_args": get_ray_init_args(cfg, hydra_cfg),
        "keep_initialised": cfg.keep_ray_initialized,
        "actor_kwargs": {"on_actor_init_fn": get_on_actor_init_fn(cfg)}
    }
    # Get remaining num of rounds if continuing a run
    num_rounds = cfg.num_rounds
    num_rounds -= strategy.last_round
    if num_rounds <= 0:
        logger.warning("Run is already completed.")
    else:
        config = ServerConfig(num_rounds=num_rounds, round_timeout=cfg.round_timeout)
        fl.simulation.start_simulation(config=config, **simulation_opts)
    history = strategy.get_history(flwr_format=True)

    # TODO: maybe stick to json, but make sure to differentiate between completed and in-progress.
    #       For now, json file is the in-progress, and pickle file is the completed.
    data = {"history": history, "cfg": cfg}
    history_path = os.path.join(save_path, HISTORY_FILE)
    with open(history_path, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # TODO(refactor): better to handle this more cleanly
    if hydra_cfg.mode == RunMode.MULTIRUN and cfg.clear_sweep_run_files:
        if "private_dir" in cfg.client:
            client_private_dir = os.path.join(save_path, cfg.client.private_dir)
            logger.info(f"Clearing private states dir: {client_private_dir}")
            shutil.rmtree(client_private_dir)
        if hasattr(strategy, "model_path") and os.path.exists(strategy.model_path):
            logger.info(f"Clearing model state: {strategy.model_path}")
            os.remove(strategy.model_path)
        if hasattr(strategy, "optimizer_path") and os.path.exists(strategy.optimizer_path):
            logger.info(f"Clearing optimizer state: {strategy.optimizer_path}")
            os.remove(strategy.optimizer_path)

    return optuna_objective(history)  # for optuna sweeps


if __name__ == "__main__":
    main()
