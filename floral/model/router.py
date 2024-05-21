import os
import glob
import torch
from torch import nn
import torch.nn.functional as F
from flwr.common.logger import logger


class Router(nn.Module):
    def __init__(self, num_clusters, noise_std=1., top2_gating=False, temp=1.0, layers=None):
        super().__init__()
        self.num_clusters = num_clusters
        self.noise_std = noise_std
        self.top2_gating = top2_gating
        self.inv_temp = 1. / (temp + 1e-10)
        self.layers = layers
        init_weight = lambda: torch.zeros(self.num_clusters)
        if self.layers is None:
            self.weight = nn.Parameter(init_weight())
        else:
            self.weight = nn.ParameterDict({
                layer_id: nn.Parameter(init_weight())
                for layer_id in self.layers
            })
        self.reset()

    def reset(self):
        self.routes = None
        self.route_probs = None

    @staticmethod
    def get_top_route(routes):
        return routes * F.one_hot(routes.argmax(-1), routes.shape[-1])

    def print_routes(self, prefix=""):
        pretty = lambda x: x.mul(100).round().int().tolist()
        if self.routes is not None:
            if self.layers is None:
                logger.debug(prefix + f"routes = {pretty(self.route_probs)}")
            else:
                for layer_id, route_probs in self.route_probs.items():
                    logger.debug(prefix + f"{layer_id} routes = {pretty(route_probs)}")

    def regularizer(self):
        # maximize entropy
        if self.route_probs is None:
            return 0.
        if self.layers is None:
            route_probs = self.route_probs[self.route_probs > 0.]
            return route_probs.mul(route_probs.log().neg()).sum()
        else:
            return sum(
                route_probs[route_probs > 0.].mul(route_probs[route_probs > 0.].log().neg()).sum()
                for route_probs in self.route_probs.values()
            )

    def moe(self, weight, eps=1e-10):
        if self.training:
            # noise_mult = 1 + 1e-2 * torch.rand_like(weight)
            # weight = weight * noise_mult
            noise_add = self.noise_std * torch.rand_like(weight)
            weight = weight + noise_add
        route_probs = weight.mul(self.inv_temp).softmax(dim=-1)

        if self.top2_gating:
            top_route = self.get_top_route(route_probs)
            if self.training:
                second_top_route = self.get_top_route(route_probs - top_route)
                top2_routes = top_route + second_top_route
                routes = top2_routes / (top2_routes.sum() + eps)
            else:
                routes = top_route / (top_route.sum() + eps)
        else:
            routes = route_probs

        return routes, route_probs

    def forward(self):
        if self.layers is None:
            self.routes, self.route_probs = self.moe(self.weight + 0.0)
        else:
            self.routes, self.route_probs = {}, {}
            for layer in self.layers:
                self.routes[layer], self.route_probs[layer] = self.moe(self.weight[layer] + 0.0)

        return self.routes

