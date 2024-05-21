import os
import torch
import numpy as np
from copy import deepcopy
from collections import defaultdict
from flwr.common.typing import Parameters, FitRes, NDArrays
from flwr.server.client_proxy import ClientProxy
from flwr.common import parameters_to_ndarrays
from floral.utils import get_ndarrays, set_ndarrays
from .fedavg_base import FedAvgBase


class MomentumClustering(FedAvgBase):
    """
    NOTE: This implementation is not suitable for practice (not that it
          would work in practice anyway). This is because the server updates
          the models directly by overwriting their private state files. Thus,
          the clients are be private as well and only train locally (or so
          they would think). I did that because I don't feel like implementing
          another strategy.
    """
    def __init__(self, *args,
                 global_model: torch.nn.Module,
                 save_path: str,
                 global_lr: float = 1.0,
                 private_dir: str = "pvt",
                 num_clusters: int = None,
                 thresholding_quantile: float = 0.2,
                 thresholding_rounds: int = 2,
                 cluster_grads=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.global_model = global_model
        self.save_path = save_path
        self.global_lr = global_lr
        self.private_dir = private_dir
        self.num_clusters = num_clusters
        self.thresholding_quantile = thresholding_quantile
        self.thresholding_rounds = thresholding_rounds
        self.cluster_grads = cluster_grads
        self.centers: dict[int, NDArrays] = {}
        self.cluster_members: dict[int, set[int]] = {}

    @torch.no_grad()
    def aggregate(self,
                  results: list[tuple[ClientProxy, FitRes]],
                  ) -> Parameters:
        # save initial global model
        global_parameters = deepcopy(get_ndarrays(self.global_model))
        # Make results into dict of ndarrays
        fit_results = {
            int(client_proxy.cid): (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for client_proxy, fit_res in results
        }
        if self.cluster_grads:
            # Make results into dict of pseudo-grads (i.e. deltas)
            fit_results = {
                cid: (self.subtract_parameters(global_parameters, parameters), num)
                for cid, (parameters, num) in fit_results.items()
            }

        # Cluster the pseudo-grads
        self.centers, self.cluster_members = self.threshold_clustering(
            fit_results,
            self.get_centers(fit_results),
            q=self.thresholding_quantile,
            thresholding_rounds=self.thresholding_rounds,
        )

        # Apply clustered pseudo-grads
        for cid in fit_results.keys():
            # Get client's cluster
            cluster_id = next(cluster_id for cluster_id in self.centers.keys()
                              if cid in self.cluster_members[cluster_id])
            if self.cluster_grads:
                delta = self.centers[cluster_id]
            else:
                delta = self.subtract_parameters(global_parameters, self.centers[cluster_id])
            personal_parameters = self.subtract_parameters(
                global_parameters, delta, alpha=self.global_lr)
            # Save personalized parameters into client's private file
            set_ndarrays(self.global_model, personal_parameters)
            model_path = os.path.join(self.save_path, self.private_dir, f"{cid}.pt")
            # NOTE: client gets parameters from here
            torch.save(self.global_model.state_dict(), model_path)

        return []  # Clients updated directly via private state file

    def get_centers(self, fit_results: dict[int, tuple[NDArrays, int]]) -> dict[int, NDArrays]:
        if len(self.centers) == 0:
            # initialize cluster memberships randomly
            first_point, _ = next(iter(fit_results.values()))
            shuffled_cids = list(fit_results.keys())
            if self.num_clusters is None:
                self.num_clusters = len(shuffled_cids)
            np.random.shuffle(shuffled_cids)
            cluster_size = int(np.ceil(len(shuffled_cids) / self.num_clusters))
            for cluster_id in range(self.num_clusters):
                cluster_indices = slice(cluster_id * cluster_size, (cluster_id + 1) * cluster_size)
                self.cluster_members[cluster_id] = set(shuffled_cids[cluster_indices])
                # Get cluster average
                cluster_avg = [np.zeros_like(layer) for layer in first_point]
                cluster_num = sum(fit_results[cid][1] for cid in self.cluster_members[cluster_id])
                for cid in self.cluster_members[cluster_id]:
                    point, num = fit_results[cid]
                    weight = num / cluster_num
                    for layer_idx in range(len(first_point)):
                        cluster_avg[layer_idx] += weight * point[layer_idx]
                self.centers[cluster_id] = cluster_avg

        return self.centers  # return previous centers, or newly initialized if none

    @staticmethod
    def subtract_parameters(parameters1: NDArrays, parameters2: NDArrays, alpha=1.0) -> NDArrays:
        return [parameters1[i] - alpha * parameters2[i] for i in range(len(parameters1))]

    # semistaticmethod
    def get_distance(self, parameters1: NDArrays, parameters2: NDArrays) -> float:
        return np.sqrt(np.sum(
            np.sum(g**2) for g in self.subtract_parameters(parameters1, parameters2)
        ))

    # semistaticmethod
    def get_dist_quantile(self,
                          fit_results: dict[int, tuple[NDArrays, int]],
                          center: NDArrays,
                          q: float = 0.2,
                          ) -> float:
        return np.quantile([
            self.get_distance(point, center) for point, _ in fit_results.values()
        ], q=q)

    # semistaticmethod
    def threshold_clustering(self,
                             fit_results: dict[int, tuple[NDArrays, int]],
                             centers: dict[int, NDArrays],
                             q: float = 0.2,
                             thresholding_rounds: int = 1,
                             ) -> dict[int, NDArrays]:
        # Threshold clustering
        dists = defaultdict(dict)
        cluster_members = defaultdict(set)
        total_num = sum(n for _, n in fit_results.values())
        for _ in range(thresholding_rounds):
            for cluster_id in range(len(centers)):
                center = centers[cluster_id]
                radius = self.get_dist_quantile(fit_results, center, q=q)
                # radius = 0.99 * min(radius, min_center_dists)
                # Average points in cluster, use center if point is further than radius
                cluster_avg = [np.zeros_like(layer) for layer in center]
                for cid, (point, num) in fit_results.items():
                    dists[cid][cluster_id] = self.get_distance(point, center)
                    weight = num / total_num
                    if dists[cid][cluster_id] <= radius:
                        # cluster_members[cluster_id].add(cid)
                        for layer_idx in range(len(center)):
                            cluster_avg[layer_idx] += weight * point[layer_idx]
                    else:
                        for layer_idx in range(len(center)):
                            cluster_avg[layer_idx] += weight * center[layer_idx]
                centers[cluster_id] = cluster_avg

        # Recalculate distances from new centers
        for cluster_id in range(len(centers)):
            for cid, (point, num) in fit_results.items():
                dists[cid][cluster_id] = self.get_distance(point, centers[cluster_id])
        # Assign point to the closest cluster center
        for cid in fit_results.keys():
            _, cluster_id = min((dist, cluster_id) for cluster_id, dist in dists[cid].items())
            cluster_members[cluster_id].add(cid)

        return centers, cluster_members
