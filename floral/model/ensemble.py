import torch
import torch.nn as nn
from floral.model import Router


class Ensemble(nn.Module):
    def __init__(self, create_model, num_clusters, router_opts={}) -> None:
        super().__init__()
        self.models = nn.ModuleList([create_model() for _ in range(num_clusters)])
        self.router = Router(num_clusters=num_clusters, **router_opts)

    def forward(self, *args):
        self.router.reset()
        routes = self.router()
        outputs = []
        for i, model in enumerate(self.models):
            if routes[i] < 1e-3:
                continue
            outputs.append(routes[i] * model(*args))
        return torch.stack(outputs).sum(0)

