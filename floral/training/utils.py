import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig
from typing import Callable, Optional
from flwr.common.logger import logger
from .trainer_base import TrainerBase
from floral.dataset.synthetic_datasets import eval_synthetic_metrics
from floral.model import Ensemble
from floral.dataset import get_dummy_input_data
from floral.floral import Floral
from torchinfo import summary


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else None

    def get_val(self):
        return self.val

    def get_avg(self):
        return self.avg


class Regularizer:
    def __init__(self, regularizers: dict[str, Callable] = {}) -> None:
        self.regularizers = regularizers
        self.last_values: dict[str, float] = {k: None for k in regularizers.keys()}

    def __call__(self, model: torch.nn.Module) -> torch.Tensor:
        device = next(model.parameters()).device  # infer device from model
        if len(self.regularizers) == 0:
            return torch.zeros(1).to(device)
        reg = torch.zeros(1).to(device)
        for name, regularizer in self.regularizers.items():
            reg_val = regularizer["function"](model)
            reg += reg_val * regularizer["parameter"]
            self.last_values[name] = reg_val.item()
        return reg

    def add_regularizer(self, name: str, parameter: float, function: Callable) -> None:
        self.regularizers[name] = {
            "function": function,
            "parameter": parameter,
        }
        self.last_values[name] = None

    def remove_regularizer(self, name: str) -> None:
        if name in self.regularizers:
            del self.regularizers[name]
        if name in self.last_values:
            del self.last_values[name]

    def as_dict(self) -> dict[str, float]:
        return dict(**self.last_values)


# TODO(refactor): put in main?
def instantiate_model(cfg: DictConfig, client_id: Optional[str] = None) -> nn.Module:
    if cfg.method.startswith("ensemble"):
        model = Ensemble(lambda: instantiate(cfg.model), **cfg.ensemble)
    else:
        model = instantiate(cfg.model)
        if cfg.method.startswith("floral"):
            model = Floral(model, **cfg.floral)

    if "router_diagonal_init" in cfg and cfg.router_diagonal_init and client_id is not None:
        router_diagonal_init_(model, client_id)
        cfg.router_lr = 0.0  # force router learning rate to be 0

    # Print floral stats
    # TODO(refactor): put in function, move to main
    if client_id is None:
        if hasattr(model, "print_stats"):
            model.print_stats()
        input_data = get_dummy_input_data(cfg)
        col_names = ["input_size", "output_size", "kernel_size", "num_params", "params_percent", "mult_adds"]
        logger.debug(summary(model, input_data=input_data, batch_dim=0, col_names=col_names, depth=10, verbose=0))

    return model


# Diagonal init of router (the optimal assignment of the synthetic dataset)
@torch.no_grad()
def router_diagonal_init_(model, client_id: str):
    try:
        int(client_id)
        client_id_is_int = True
    except ValueError:
        client_id_is_int = False

    if not client_id_is_int:
        logger.warning("'router_diagonal_init' is true but client ids are not integers.")

    elif not hasattr(model, "router"):
        logger.warning("'router_diagonal_init' is true but model does not have a router.")

    else:
        cluster_id = int(client_id) % len(model.router.weight)
        with torch.no_grad():
            # After softmax, ~1 at cluster_id and ~0 everywhere else
            model.router.weight -= 100.
            model.router.weight[cluster_id] = 0.


def get_param_groups(cfg, model):
    if cfg.method.startswith("floral"):
        return [
            {
                "params": model.base_model.parameters(),
                "name": "base_model",
            },
            {
                "params": model.lora_modules.parameters(),
                "name": "lora_modules",
                "lr": cfg.lora_lr,
            },
            {
                "params": model.router.parameters(),
                "name": "router",
                "lr": cfg.router_lr,
                "weight_decay": 0.0,  # don't decay router
            },
        ]
    elif cfg.method.startswith("ensemble"):
        return [
            {
                "params": model.models.parameters(),
                "name": "models",
            },
            {
                "params": model.router.parameters(),
                "name": "router",
                "lr": cfg.router_lr,
                "weight_decay": 0.0,  # don't decay router
            },
        ]
    else:
        return model.parameters()


# TODO(refactor): I don't like this function...
def get_custom_eval_fn(cfg: OmegaConf, cid: str) -> Callable[..., dict[str, float]]:
    @torch.no_grad()
    def custom_eval_fn(trainer: TrainerBase, output: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
        if cfg.task.startswith("synthetic"):
            # NOTE: cid should be castable to int (should be true by dataset construction)
            dataset = trainer.dataloaders['train'].dataset.dataset.fl_dataset
            return eval_synthetic_metrics(dataset, trainer.model, int(cid))

        elif cfg.task.startswith("stackoverflow"):
            from floral.dataset.tff_datasets.stackoverflow import (
                so_metrics_of_batch_fn, get_special_tokens)
            special_tokens = get_special_tokens(cfg.dataset.vocab_size, cfg.dataset.num_oov_buckets)
            non_vocab_idx = [special_tokens.pad, *special_tokens.oov, special_tokens.bos, special_tokens.eos]
            _, metrics = so_metrics_of_batch_fn(output, target, non_vocab_idx=non_vocab_idx)
            return metrics

        elif cfg.task.startswith("shakespeare"):
            from floral.dataset.tff_datasets.shakespeare import shakespeare_metrics_of_batch_fn
            _, metrics = shakespeare_metrics_of_batch_fn(output, target)
            return metrics

        else:
            return {"acc": accuracy(output, target).item()}

    return custom_eval_fn


def accuracy(output, target):
    _, predicted = torch.max(output, 1)
    correct = (predicted == target).float()
    acc = correct.mean().mul(100.)
    return acc


def get_router_regularizer():
    def router_regularizer(model):
        return model.router.regularizer()

    return router_regularizer

