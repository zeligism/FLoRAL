import os
import shutil
import glob
import flwr as fl
from pathlib import Path
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from logging import getLevelName
from flwr.client.typing import ClientFn
from flwr.common.logger import logger
from flwr.client import Client
from hydra.utils import instantiate
from floral.client import FlowerClientWithPrivateModules
from floral.utils import init_seed
from floral.training.utils import (
    instantiate_model,
    get_param_groups,
    get_custom_eval_fn,
)


def configure_actor_logger(logdir: Path, loglevel: str = "INFO", identifier: str = "", max_logfiles: int = 1000) -> None:
    fl.common.logger.update_console_handler(level=getLevelName(loglevel), timestamps=True, colored=True)
    # Create actor logfile only if no handlers exist
    filename = os.path.join(logdir, f"p{os.getpid()}.log")
    file_handlers = [h for h in logger.handlers if hasattr(h, "baseFilename")]
    logfile_handler_exists = any(".log" in h.baseFilename for h in file_handlers)
    num_actor_logfiles = len(glob.glob(os.path.join(logdir, "p*.log")))  # excluding main.log
    if not logfile_handler_exists and num_actor_logfiles < max_logfiles:
        # NOTE: some clients can simultaneously satisfy this check and proceed at creating a logfile,
        #       introducing a sort of a race condition that result in 'num_actor_logfiles' being
        #       larger than 'max_logfiles' (but not much larger, so it's not a big deal).
        fl.common.logger.configure(identifier=identifier, filename=filename)


def get_on_actor_init_fn(cfg):
    def on_actor_init_fn():
        from .. import _meta_init
        _meta_init()

    return on_actor_init_fn


def split_clients(clients_ids, unseen_clients=0., seed=0):
    # Use integers as clients_ids for the simulator
    int_clients_ids = list(clients_ids.keys())
    unseen_int_clients_ids = []
    if unseen_clients > 0:
        # For testing generalization to unseen clients
        assert unseen_clients < len(int_clients_ids)
        import random
        if unseen_clients < 1:
            num_unseen = round(unseen_clients * len(int_clients_ids))
        else:
            num_unseen = round(unseen_clients)
        # Create a reproducible split
        random.Random(seed).shuffle(int_clients_ids)
        unseen_int_clients_ids = list(sorted(int_clients_ids[-num_unseen:]))
        int_clients_ids = list(sorted(int_clients_ids[:-num_unseen]))

    return int_clients_ids, unseen_int_clients_ids


def get_client_fn(
        cfg: DictConfig,
        client_data: list[tuple[DataLoader, DataLoader]],
        save_path: Path,
        client_mode = "train",  # TODO: remove
        local_client=False,  # TODO: remove
        ) -> tuple[ClientFn, dict[str, str]]:
    
    # NOTE: flwr simulation does not seem to support non-int client ids (is this a bug?)
    #       E.g., check `_wrap_recordset_in_message` in flwr/simulation/ray_transport/ray_client_proxy.py.
    #       Thus, non-int `clients_ids` are sorted, and their orders are used as the ids instead.
    #       The order is unique for a FIXED pool of clients, so it might not be consistent across runs.
    try:
        # If cid is int, then just use it directly
        clients_ids = {str(int(cid)): cid for cid in client_data.keys()}
    except ValueError:
        # If we are here, then some cids are non-int, so use order instead as the cid for flwr simulation
        clients_ids = {str(i): cid for i, cid in enumerate(sorted(client_data.keys()))}

    # Initialize private dir and clear it if starting a fresh training run
    if "private_dir" in cfg.client:
        private_dir = os.path.join(save_path, cfg.client.private_dir)
        if not cfg.continue_training and os.path.exists(private_dir):
            logger.debug(f"Removing stale private dir '{private_dir}'")
            shutil.rmtree(private_dir)
        os.makedirs(private_dir, exist_ok=True)


    def client_fn(int_cid: str) -> Client:
        cid = clients_ids[int_cid]  # retrieve the original cid

        configure_actor_logger(cfg.logdir, identifier=cfg.identifier, max_logfiles=cfg.max_logfiles)

        if cfg.deterministic:
            init_seed(cfg.seed)
        model = instantiate_model(cfg, cid)
        optimizer = instantiate(cfg.optimizer, get_param_groups(cfg, model))
        dataloaders = client_data[cid]
        # tff dataloaders are handled differently due to serialization issues
        if "_tff" in dataloaders:
            tff_datasets = dataloaders["_tff"]
            dataloaders = {split: tff_datasets[split].get_client_dataloader(cid)
                           for split in dataloaders.keys() if split != "_tff"}

        trainer = instantiate(
            cfg.trainer,
            model=model,
            optimizer=optimizer,
            dataloaders=dataloaders,
            identifier=f"Client {cid}",
            custom_eval_fn=get_custom_eval_fn(cfg, cid),
        )
        client = instantiate(
            cfg.client,
            trainer=trainer,
            save_path=save_path,
            client_id=cid,
        )
        if isinstance(client, FlowerClientWithPrivateModules):
            client.add_private_modules(cfg.extra_private_modules)

        return client.to_client()


    return client_fn, clients_ids
