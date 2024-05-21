
import torch
from collections import defaultdict
from typing import Any

from floral.model import Ensemble
from .trainer import Trainer
from .utils import AverageMeter
from flwr.common.logger import logger


class EMTrainer(Trainer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert 'val' in self.dataloaders, "EM Trainer requires a validation dataset"
        # model passed is an ensemble, which we explicitly denote as `self.ensemble`
        self.ensemble: Ensemble = self.model
        # `sample_weights` assigns a weight for each data sample
        C, N_i = len(self.ensemble.models), len(self.dataloaders['val'])
        self.samples_weights = torch.ones(C, N_i).div(C)

    def ensemble_step(self, mode: str, batch: Any) -> dict[str, float]:
        """
        Wraps the regular train/eval step by calling ensemble with one model active at a time.
        """
        weighted_avg_metric = defaultdict(float)
        router_weight = self.ensemble.router.weight.clone().detach()
        model_weights = router_weight.softmax(-1).tolist()
        for model_idx, _ in enumerate(self.ensemble.models):
            # Activate only one model at a time
            with torch.no_grad():
                self.ensemble.router.weight.sub_(100.)
                self.ensemble.router.weight[model_idx] = 0.0
            # Run a step of whatever
            if mode == 'train':
                metrics = super().train_step(batch)
            else:
                metrics = super().eval_step(batch)
            # Weight metrics accordingly
            for k, v in metrics.items():
                weighted_avg_metric[k] += model_weights[model_idx] * v
        # reset router weight
        with torch.no_grad():
            self.ensemble.router.weight.copy_(router_weight)
        return weighted_avg_metric

    def train_step(self, batch: Any) -> dict[str, float]:
        return self.ensemble_step('train', batch)

    def eval_step(self, batch: Any) -> dict[str, float]:
        return self.ensemble_step('eval', batch)

    def train(self) -> dict[str, AverageMeter]:
        self.router_em_()
        metrics_meter = super().train()
        return metrics_meter

    @torch.no_grad()
    def router_em_(self):
        # Gather validation losses
        # val_losses = torch.zeros(len(self.ensemble.models), len(self.dataloaders['val']))  # XXX
        val_losses = []
        for model in self.ensemble.models:
            model_val_losses = []
            for batch in self.dataloaders['val']:
                data, target = self.batch_preprocess(batch)
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                val_loss = self.loss_fn(output, target)
                model_val_losses.append(val_loss.view(-1))
            val_losses.append(torch.cat(model_val_losses))
        val_losses = torch.stack(val_losses)
        # S_i <- softmax(w - S_i, C)
        # w <- S_i.mean(N_i)
        self.samples_weights = torch.softmax(self.ensemble.router.weight.view(-1,1) - val_losses, dim=0)
        self.ensemble.router.weight.copy_(self.samples_weights.mean(dim=1))

    # @staticmethod
    # @torch.no_grad()
    # def router_em_(model, get_loss):
    #     router_weight_old = model.router.weight.clone()
    #     router_weight = router_weight_old.clone()
    #     for c in range(len(model.router.weight)):
    #         w = torch.zeros_like(model.router.weight).sub(100.)
    #         w[c] = 0.0
    #         model.router.weight.copy_(w)
    #         loss = get_loss()
    #         router_weight[c].sub_(loss, alpha=1.0)
    #     log_Z = router_weight.logsumexp(0) - router_weight_old.logsumexp(0)
    #     model.router.weight.copy_(router_weight.sub(log_Z))
    #     # model.router.router_probs_queue.append(torch.softmax(router_weight, dim=0))
    #     # model.router.weight.copy_(router_weight_old)
