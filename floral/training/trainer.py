
import torch
from collections import defaultdict
from math import ceil
from typing import Callable, Iterable, Optional, Any
from flwr.common.logger import logger

from floral.utils import init_device
from .utils import Regularizer, AverageMeter

DEFAULT_BATCH_KEYS = ["image", "label"]


class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 dataloaders: dict[str, Iterable],
                 loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 regularizer: Regularizer = Regularizer(),
                 identifier: str = "Client",
                 local_epochs: float = 1,
                 batch_keys: list[str] = DEFAULT_BATCH_KEYS,
                 custom_eval_fn: Callable[..., dict[str, float]] = lambda *_: {},
                 eval_only: bool = False,
                 clip_grad_norm: Optional[float] = None,
                 ) -> None:
        self.device = init_device()
        # self.device = torch.device("cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.dataloaders = dataloaders
        self.loss_fn = loss_fn
        self.identifier = identifier
        self.regularizer = regularizer
        self.local_epochs = local_epochs
        self.batch_keys = batch_keys
        self.custom_eval_fn = custom_eval_fn
        self.eval_only = eval_only  # TODO
        self.clip_grad_norm = clip_grad_norm

    def report_and_update(self,
                          metrics: dict[str, float],
                          metrics_meter: dict[str, AverageMeter],
                          prefix: str = "") -> None:
        report_fields = []
        for k, v in metrics.items():
            metrics_meter[k].update(v)
            report_fields.append(f"{k}={metrics_meter[k].get_val():.4f}")
        report = prefix + "\t".join(report_fields)
        logger.info(report)

    def stopping_criterion(self, epoch: int, batch_idx: int) -> bool:
        return epoch + (batch_idx + 1) / len(self.dataloaders['train']) >= self.local_epochs

    def batch_preprocess(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(batch, dict):
            batch = [batch[k] for k in self.batch_keys]
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            data, target = batch
        else:
            raise NotImplementedError(batch)
        return data, target

    def train(self, config: dict[str, Any]) -> dict[str, float]:
        self.model.train()
        metrics_meter = defaultdict(AverageMeter)
        for epoch in range(ceil(self.local_epochs)):
            for batch_idx, batch in enumerate(self.dataloaders['train']):
                metrics = self.train_step(batch)
                prefix = f"Train | {self.identifier}: " + \
                        f"[{batch_idx+1}/{len(self.dataloaders['train'])}] "
                self.report_and_update(metrics, metrics_meter, prefix=prefix)
                if self.stopping_criterion(epoch, batch_idx):
                    break

        return {k: meter.get_avg() for k, meter in metrics_meter.items()}

    @torch.no_grad()
    def evaluate(self, config: dict[str, Any]) -> dict[str, float]:
        self.model.eval()
        metrics_meter = defaultdict(AverageMeter)
        for batch_idx, batch in enumerate(self.dataloaders['test']):
            metrics = self.eval_step(batch)
            prefix = f"Test | {self.identifier}: " + \
                     f"[{batch_idx+1}/{len(self.dataloaders['test'])}] "
            self.report_and_update(metrics, metrics_meter, prefix=prefix)

        return {k: meter.get_avg() for k, meter in metrics_meter.items()}

    def train_step(self, batch: Any) -> dict[str, float]:
        data, target = self.batch_preprocess(batch)
        data, target = data.to(self.device), target.to(self.device)
        self.optimizer.zero_grad()
        output = self.model(data)
        loss = self.loss_fn(output, target).mean()
        reg = self.regularizer(self.model).to(self.device)
        (loss + reg).backward()
        if self.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
        self.optimizer.step()

        return {"loss": loss.item(), **self.regularizer.as_dict()}

    def eval_step(self, batch: Any) -> dict[str, float]:
        data, target = self.batch_preprocess(batch)
        data, target = data.to(self.device), target.to(self.device)
        output = self.model(data)
        loss = self.loss_fn(output, target).mean()
        custom_eval_metrics = self.custom_eval_fn(self, output, target)

        return {"loss": loss.item(), **custom_eval_metrics}
