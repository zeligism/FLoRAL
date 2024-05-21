from torch.utils.data import DataLoader
from flwr_datasets import FederatedDataset
from .mnist_cifar import (
    define_cluster_transforms_getter,
    get_apply_transforms,
    get_partitioners,
)
FLWR_DATASETS = ("mnist", "cifar10", "cifar100")


def get_flwr_data(dataset="mnist",
                  subtask="",
                  num_clusters=4,
                  num_clients=20,
                  num_classes=10,
                  image_key="image",
                  label_key="label",
                  batch_size=32,
                  test_batch_size=128,
                  val_proportion=0.0,
                  train_reduction=0.0,
                  min_reduced_size=10,
                  dataloader_cfg={},
                  ) -> dict[str, dict[str, DataLoader]]:
    assert dataset in FLWR_DATASETS
    batch_keys = [image_key, label_key]

    # This is how we define the ground truth cluster (made explicit for clarity)
    def get_cluster(client_id):
        return client_id % num_clusters

    # This function gets the specific transform for each cluster and split
    get_transforms = define_cluster_transforms_getter(
        dataset, subtask, num_classes, num_clusters)
    # The partitioning is defined based on the task (i.e. dataset)
    partitioners = get_partitioners(
        dataset, num_clients, num_clusters, *batch_keys)
    # Use flwr (ultimately, huggingface) to fetch and partition the data
    fds = FederatedDataset(dataset=dataset, partitioners=partitioners)
    # Just retrieve the clients' data loaders with their cluster transformations
    data_loaders = {}  # modes
    for client_id in range(num_clients):
        cluster_id = get_cluster(client_id)
        train_dataset = fds.load_partition(client_id, "train").with_transform(
            get_apply_transforms(*get_transforms(cluster_id, "train"), *batch_keys)
        )
        if train_reduction > 0.0:
            reduced_train_size = len(train_dataset) * (1 - train_reduction)
            if reduced_train_size < min_reduced_size:
                train_reduction = len(train_dataset) - min_reduced_size
            train_dataset = train_dataset.train_test_split(test_size=train_reduction)["train"]
        if val_proportion > 0.0:
            ds = train_dataset.train_test_split(test_size=val_proportion, shuffle=True)
            train_dataset, val_dataset = ds["train"], ds["test"]
        test_dataset = fds.load_partition(client_id, "test").with_transform(
            get_apply_transforms(*get_transforms(cluster_id, "test"), *batch_keys)
        )

        data_loaders[str(client_id)] = {
            "train": DataLoader(train_dataset, batch_size=batch_size, **dataloader_cfg),
            "test": DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, **dataloader_cfg),
        }
        if val_proportion > 0.0:
            data_loaders[str(client_id)]["val"] = \
                DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False, **dataloader_cfg)

    return data_loaders
