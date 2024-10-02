import torch
from typing import Callable, Iterable, Any
# from .utils import Regularizer  # TODO: causes circular import: fix


class TrainerBase:

    model: torch.nn.Module
    dataloaders: dict[str, Iterable]
    optimizer: torch.optim.Optimizer
    loss_fn: Callable
    device: torch.device
    # regularizer: Regularizer
    identifier: str
    local_epochs: float
    batch_keys: list[str]
    custom_eval_fn: Callable[..., dict[str, float]]

    def report_and_update(self, *args, **kwargs) -> None:
        ...

    def stopping_criterion(self, *args, **kwargs) -> bool:
        ...

    def batch_preprocess(self, batch: Any) -> Any:
        ...

    def train(self, config: dict[str, Any]) -> dict[str, float]:
        ...

    def evaluate(self, config: dict[str, Any]) ->  dict[str, float]:
        ...

    def train_step(self, batch: Any) -> dict[str, float]:
        ...

    def eval_step(self, batch: Any) -> dict[str, float]:
        ...
