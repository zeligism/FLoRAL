
import os
import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class SyntheticDatasetClient(Dataset):
    def __init__(self, fl_dataset, client_id=None):
        self.fl_dataset = fl_dataset
        self.set_client(client_id)

    def set_client(self, index=None):
        fl = self.fl_dataset
        if index is None:
            self.client_id = None
            self.data = torch.flatten(fl.data, start_dim=0, end_dim=1)
            self.targets = torch.flatten(fl.targets, start_dim=0, end_dim=1)
            self.length = len(self.data)
        else:
            if index < 0 or index >= fl.num_clients:
                raise ValueError('Number of clients is out of bounds.')
            self.client_id = index
            self.data = fl.data[index]
            self.targets = fl.targets[index]
            self.length = len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return self.length


class SyntheticDataset:
    def __init__(self, data_dir=None, num_clients=10, samples_per_client=10, num_clusters=3,
                 seed=0, dim=10, dim_out=3, rank=1, uv_constant=1.0, label_noise_std=0.01,
                 func=torch.nn.Identity(), normalize_hidden=False,
                 simple=False, linear=False, classify=False, load=False):
        self.data_dir = data_dir
        self.num_clients = num_clients
        self.num_clusters = num_clusters
        self.seed = seed
        self.samples_per_client = samples_per_client
        self.func = func
        self.dim = dim
        self.dim_out = dim_out
        self.rank = rank
        self.uv_constant = uv_constant
        self.label_noise_std = label_noise_std
        self.normalize_hidden = normalize_hidden
        self.simple = simple
        self.linear = linear
        self.load = load
        self.classify = classify  # whether to classify or to regress
        self.generate_dataset_parameters()
        self.sample_data()

    @torch.no_grad()
    def generate_dataset_parameters(self):
        g = torch.Generator().manual_seed(self.seed + 27)
        if self.simple:
            W = torch.eye(self.dim).unsqueeze(0)
            us = torch.zeros(self.num_clusters, self.dim, self.rank)
            vs = torch.zeros(self.num_clusters, self.rank, self.dim)
            for r in range(self.rank):
                for c in range(self.num_clusters):
                    us[c, c, r] = 1.
                    vs[c, r, c+r] = 1.
            biases = torch.zeros(1, 1, self.dim)
            W_out = torch.ones(self.dim, self.dim_out).unsqueeze(0) / self.dim ** 0.5
        else:
            W = torch.randn(1, self.dim, self.dim, generator=g) / self.dim ** 0.5
            us = torch.randn(self.num_clusters, self.dim, self.rank, generator=g) / self.dim ** 0.5
            vs = torch.randn(self.num_clusters, self.rank, self.dim, generator=g) / self.rank ** 0.5
            biases = torch.randn(1, 1, self.dim, generator=g) / self.dim ** 0.5
            W_out = torch.randn(1, self.dim, self.dim_out, generator=g) / self.dim ** 0.5

        # Add `num_cluster` low rank differences across clients, interleaved
        uvs = self.uv_constant * us.matmul(vs).repeat(self.num_clients // self.num_clusters, 1, 1)
        Ws = W.add(uvs)

        self.W = W
        self.Ws = Ws
        self.W_out = W_out
        self.biases = biases

        if self.data_dir is not None:
            filename = ",".join([f"synthetic{'_linear' if self.linear else ''}{'_simple' if self.simple else ''}",
                                 f"s={self.seed}",
                                 f"n={self.samples_per_client}",
                                 f"K={self.num_clients}",
                                 f"C={self.num_clusters}",
                                 f"r={self.rank}",
                                 f"d={self.dim}", f"{self.dim_out}",
                                 f"uv={self.uv_constant}",
                                 f"f={self.func}",
                                 ]) + ".pkl"
            synthetic_dir = os.path.join(self.data_dir, "synthetic")
            dataset_path = os.path.join(synthetic_dir, filename)
            if self.load and os.path.exists(dataset_path):
                with open(dataset_path, "rb") as f:
                    (self.W, self.Ws, self.W_out, self.biases) = pickle.load(f)
            else:
                os.makedirs(synthetic_dir, exist_ok=True)
                with open(dataset_path, "wb") as f:
                    data = (self.W, self.Ws, self.W_out, self.biases)
                    pickle.dump(data, f)

    @torch.no_grad()
    def sample_data(self):
        g = torch.Generator().manual_seed(self.seed + 11)
        # y = f(XW) W_out (= XW' for some W' if f = id)
        xs = torch.randn(self.num_clients, self.samples_per_client, self.dim, generator=g)
        hs = xs.matmul(self.Ws).add(self.biases)

        if self.linear:
            ys = hs[:, :, :self.dim_out]
        else:
            hs: torch.Tensor = self.func(hs)
            if self.normalize_hidden:
                # normalize coordinate-wise per client
                hs = (hs - hs.mean(1, keepdim=True)) / hs.std(1, keepdim=True)
            ys = hs.matmul(self.W_out)

        if self.label_noise_std > 0:
            ys.add_(torch.randn_like(ys, generator=g).mul(self.label_noise_std))

        self.data = xs
        self.targets = ys
        
        if self.classify:
            self.targets = F.one_hot(self.targets.argmax(-1), self.targets.shape[-1]).to(self.targets)

