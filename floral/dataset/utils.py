from omegaconf import OmegaConf
from .synthetic_datasets import get_synthetic_data
from .flwr_datasets import get_flwr_data
from .tff_datasets import get_tff_data

# Task PREFIXES
SYNTHETIC_TASKS = ("synthetic",)
FLWR_TASKS = ("mnist", "cifar")
TFF_TASKS = ("emnist", "shakespeare", "stackoverflow")


def startswithany(name: str, prefixes: tuple[str]) -> bool:
    return any(name.startswith(prefix) for prefix in prefixes)


def get_data(cfg: OmegaConf):
    if startswithany(cfg.task, SYNTHETIC_TASKS):
        return get_synthetic_data(cfg)
    elif startswithany(cfg.task, FLWR_TASKS):
        return get_flwr_data(**cfg.dataset)
    elif startswithany(cfg.task, TFF_TASKS):
        return get_tff_data(**cfg.dataset)
    else:
        raise NotImplementedError(cfg.task)


def get_dummy_input_data(cfg, b=1, seq_len=20):
    import torch
    import torch.nn.functional as F
    if startswithany(cfg.task, SYNTHETIC_TASKS):
        return torch.randn(b, cfg.dataset.dim)
    elif cfg.task.startswith("mnist") or cfg.task.startswith("emnist"):
        return torch.randn(b, 1, 28, 28)
    elif cfg.task.startswith("cifar10"):  # note that it includes cifar100
        return torch.randn(b, 3, 32, 32)
    elif cfg.task.startswith("shakespeare") or cfg.task.startswith("stackoverflow"):
        return torch.randint(high=cfg.model.vocab_size, size=(b, seq_len))
    else:
        raise NotImplementedError(cfg.task)
