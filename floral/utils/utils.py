import os
import glob
import time
import timeit
import torch
import numpy
import random
from typing import Any
from copy import deepcopy
from math import isnan
from hydra.utils import instantiate
from flwr.common.logger import logger
from flwr.common.typing import NDArrays
try:
    import wandb
    WANDB_OK = True
except ModuleNotFoundError:
    WANDB_OK = False


def init_device(try_mps=False):
    maybe_mps = "mps" if try_mps and torch.backends.mps.is_available() else "cpu"
    maybe_cuda = "cuda" if torch.cuda.is_available() else maybe_mps
    return torch.device(maybe_cuda)


def init_seed(seed):
    if isinstance(seed, (list, tuple)):
        logger.info(f"Got combined seed: {seed}")
        rng = numpy.random.default_rng(seed)  # can handle a list of seeds appropriately
        seed = int(rng.integers(2**32-1))  # reset seed
    # seed is an integer now
    logger.info(f"Setting seed to {seed}")
    if torch.cuda.device_count():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)


def now(how="timeit"):
    return timeit.default_timer() if how == "timeit" else time.time()


def _clean_pytorch_files(dir):
    for path in glob.glob(os.path.join(dir, '*.pt')):
        logger.debug(f"Removing stale private file '{path}'")
        os.remove(path)


def setup_wandb(cfg):
    cfg.wandb = WANDB_OK and cfg.wandb
    if cfg.wandb:
        wandb.init(
            project=cfg.experiment,
            config=vars(cfg)
        )


def init_private_dir(private_dir):
    # clean private directory from .pt files to avoid conflicting runs
    os.makedirs(private_dir, exist_ok=True)
    _clean_pytorch_files(private_dir)


def get_ray_init_args(cfg, hydra_cfg):
    ray_init_args = instantiate(cfg.ray_init_args)
    # configure ray reseources based on environment
    if "submitit_launcher" in hydra_cfg.launcher._target_:
        if hydra_cfg.launcher.cpus_per_task is not None:
            ray_init_args["num_cpus"] = hydra_cfg.launcher.cpus_per_task
        if hydra_cfg.launcher.gpus_per_task is not None:
            ray_init_args["num_gpus"] = hydra_cfg.launcher.gpus_per_task
    # Override with SLURM env variables if they exist
    if "SLURM_CPUS_PER_TASK" in os.environ:
        ray_init_args["num_cpus"] = int(os.environ["SLURM_CPUS_PER_TASK"])
    if "SLURM_GPUS_PER_TASK" in os.environ:
        ray_init_args["num_cpus"] = int(os.environ["SLURM_GPUS_PER_TASK"])
    # TODO:
    # SLURM_MEM_PER_CPU
    # SLURM_MEM_PER_GPU
    # SLURM_MEM_PER_NODE

    logger.info(f"Ray init args = {ray_init_args}")
    return ray_init_args


def optuna_objective(history):
    if history is not None and len(history.losses_distributed) > 0:
        _, loss = history.losses_distributed[-1]
        if isnan(loss):
            loss = float('inf')
    else:
        loss = float('inf')

    return loss


def eval_num(expression: str) -> Any:
    original_expression = expression
    for op in ['+', '-', '*', '/']:
        expression = expression.replace(op, ' ')
    try:
        for digit in expression.split():
            float(digit)
    except ValueError:
        from omegaconf.errors import InterpolationResolutionError
        raise InterpolationResolutionError(
            "Got invalid arithmetic expression: " + original_expression)

    return eval(original_expression)


def _split_strings_by_subsets(strings: list[str], subsets: list[str]):
    with_subsets = []
    without_subsets = []
    for string in strings:
        if any(subset in string for subset in subsets):
            with_subsets.append(string)
        else:
            without_subsets.append(string)
    return with_subsets, without_subsets


def _split_dict_by_subkeys(source_dict: dict[str, Any], subkeys: list[str] = []):
    with_subkeys, _ = _split_strings_by_subsets(source_dict.keys(), subkeys)
    split_dict = {}
    for k in with_subkeys:
        split_dict[k] = source_dict[k]
        del source_dict[k]
    return source_dict, split_dict


# an alias that doesn't change state_dict inplace and casts split modules as a list.
def split_state_dict(state_dict: dict[str, Any], split_modules=set()):
    return _split_dict_by_subkeys(deepcopy(state_dict), subkeys=list(split_modules))


def dict_to_ndarrays(state_dict: dict[str, Any]) -> NDArrays:
    return [val.cpu().numpy() if val is not None else None for _, val in state_dict.items()]


def ndarrays_to_dict(keys: list[str], parameters: NDArrays) -> dict[str, torch.Tensor]:
    return {k: torch.tensor(v) if v is not None else None for k, v in zip(keys, parameters)}


def get_ndarrays(model: torch.nn.Module) -> NDArrays:
    return dict_to_ndarrays(model.state_dict())


def set_ndarrays(model: torch.nn.Module, parameters: NDArrays) -> None:
    model.load_state_dict(ndarrays_to_dict(model.state_dict().keys(), parameters), strict=True)

