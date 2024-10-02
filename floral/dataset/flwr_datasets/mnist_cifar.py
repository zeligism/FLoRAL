from functools import partial
import torchvision.transforms as T
from flwr.common.logger import logger
from .natural_id_cluster_partitioner import NaturalIdClusterPartitioner


def identity(y):
    return y


def label_shift(cluster_id, num_classes, y):
    return (y + cluster_id) % num_classes


def _get_data_augmentations(dataset, mode="train"):
    if mode == "test":
        return []
    elif dataset == "mnist":
        return []
    elif dataset in ("cifar10", "cifar100"):
        return [
            T.RandomHorizontalFlip(),
            T.RandomCrop(32, padding=4),
        ]
    else:
        return []


def _get_normalization(dataset):
    if dataset == "mnist":
        return []
    elif dataset in ("cifar10", "cifar100"):
        return [
            T.Normalize((0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010)),
        ]
    else:
        return []


class ClusterTransformGetter:
    def __init__(self, dataset, subtask, num_classes, num_clusters) -> None:
        self.dataset = dataset
        self.subtask = subtask
        self.num_classes = num_classes
        self.num_clusters = num_clusters

    def __call__(self, *args, **kwargs):
        if self.subtask == "rotate":
            return self._get_transforms_rotate(*args, **kwargs)

        elif self.subtask == "label_shift":
            return self._get_transforms_label_shift(*args, **kwargs)

        else:
            return self._get_transforms_normal(*args, **kwargs)

    def _get_transforms_rotate(self, cluster_id, mode="train"):
        angle = (360 // self.num_clusters) * cluster_id
        transforms = T.Compose([
            *_get_data_augmentations(self.dataset, mode),
            T.ToTensor(),
            partial(T.functional.rotate, angle=angle),
            *_get_normalization(self.dataset),
        ])

        return transforms, identity

    def _get_transforms_label_shift(self, cluster_id, mode="train"):
        transforms = T.Compose([
            *_get_data_augmentations(self.dataset, mode),
            T.ToTensor(),
            *_get_normalization(self.dataset),
        ])

        return transforms, partial(label_shift, cluster_id, self.num_classes)

    def _get_transforms_normal(self, cluster_id, mode="train"):
        transforms = T.Compose([
            *_get_data_augmentations(self.dataset, mode),
            T.ToTensor(),
            *_get_normalization(self.dataset),
        ])

        return transforms, identity


class TransformsApplier:
    def __init__(self, transforms, label_transforms, image_key, label_key) -> None:
        self.transforms = transforms
        self.label_transforms = label_transforms
        self.image_key = image_key
        self.label_key = label_key

    def __call__(self, batch):
        batch[self.image_key] = [self.transforms(image) for image in batch[self.image_key]]
        batch[self.label_key] = [self.label_transforms(label) for label in batch[self.label_key]]
        return batch


def define_cluster_transforms_getter(*args):
    return ClusterTransformGetter(*args)


def get_apply_transforms(*args):
    return TransformsApplier(*args)


def get_partitioners(dataset, num_clients, num_clusters, image_key, label_key):
    if dataset in ("mnist", "cifar10"):
        # iid partition, transform based on cluster id (see below)
        partitioners = {"train": num_clients, "test": num_clients}

    elif dataset == "cifar100":
        # partition based on coarse label
        num_clients_per_cluster = num_clients // num_clusters
        clipped_num_clients = num_clients_per_cluster * num_clusters
        if num_clients != clipped_num_clients:
            logger.warning(f"Clipping num_clients to {clipped_num_clients} to ensure uniform cluster sizes.")
        partitioner_args = {
            "num_clusters": num_clusters,
            "num_nodes_per_cluster": num_clients_per_cluster,
            "partition_by": label_key,
        }
        natural_id_cluster_partitioner = NaturalIdClusterPartitioner(**partitioner_args)
        partitioners = {
            "train": natural_id_cluster_partitioner,
            "test": natural_id_cluster_partitioner,
        }

    return partitioners
