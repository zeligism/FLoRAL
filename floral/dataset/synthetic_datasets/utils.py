import torch
from collections import defaultdict
from torch.utils.data import DataLoader, random_split
from hydra.utils import instantiate
from .synthetic import SyntheticDataset, SyntheticDatasetClient


def get_synthetic_data(cfg) -> dict[str, dict[str, DataLoader]]:
    # Determine number of samples per client such that:
    # 1) Each client has a little data relative to model size, and
    # 2) All clients collectively have lots of data such that collab/FL helps.
    # 'cfg.client_data_to_param_ratio' controls this.
    dummy_model = instantiate(cfg.model)  # NOTE: ignores floral/ensemble instantiations = counts less params
    num_params = sum(p.numel() for p in dummy_model.parameters())
    cfg.dataset.samples_per_client = round(cfg.client_data_to_param_ratio * num_params)
    fl_dataset = instantiate(cfg.dataset)

    client_data = defaultdict(dict)
    train_test_split = (cfg.train_proportion, 1 - cfg.train_proportion)
    train_val_split = (1 - cfg.val_proportion, cfg.val_proportion)
    for cid in range(cfg.dataset.num_clients):
        client_dataset = SyntheticDatasetClient(fl_dataset, cid)
        train_dataset, test_dataset = random_split(client_dataset, train_test_split)
        client_data[str(cid)]['train'] = DataLoader(train_dataset, batch_size=cfg.batch_size, **cfg.dataloader)
        client_data[str(cid)]['test'] = DataLoader(test_dataset, batch_size=cfg.test_batch_size, shuffle=False, **cfg.dataloader)
        if cfg.val_proportion > 0:
            train_dataset, val_dataset = random_split(train_dataset, train_val_split)
            client_data[str(cid)]['val'] = DataLoader(val_dataset, batch_size=cfg.test_batch_size, shuffle=False, **cfg.dataloader)

    return client_data


@torch.no_grad()
def eval_synthetic_metrics(dataset: SyntheticDataset,
                           model: torch.nn.Module,
                           client_id: int
                           ) -> dict[str, float]:
    return {}

    # XXX: used to work for lora experts class, but it's not a priority to fix it now

    if not hasattr(model, "base_model"):
        return {}

    W, Ws = dataset.W, dataset.Ws[:dataset.num_clusters]
    uvs = Ws - W
    m = model.base_model
    cluster_probs = model.router.route_probs

    if hasattr(m, 'featurizer'):
        return {}
        m = m.featurizer[0]
        W_error = torch.zeros(1)
        uv_error = torch.zeros(1)
    else:
        if model.router.layers is not None:
            cluster_probs = cluster_probs["/"]
        W = W[0, :, :dataset.dim_out].T
        uvs = uvs[:, :, :dataset.dim_out].transpose(1, 2)
        uv = uvs[client_id % dataset.num_clusters]
        W_hat = m.weight
        uvs_hat = model.lora_modules[model.get_ref(m)].fuse()
        uv_hat = uvs_hat.mul(cluster_probs.view(-1, 1, 1)).sum(0)
        W_error = (W_hat - W.to(W_hat)).pow(2).mean()
        uv_error = (uv_hat - uv.to(uv_hat)).pow(2).mean()

    return {
        "W_error": W_error.item(),
        "uv_error": uv_error.item(),
    }
