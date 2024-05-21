import torchvision.transforms as T
from flwr.common.logger import logger
from .natural_id_cluster_partitioner import NaturalIdClusterPartitioner


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


def define_cluster_transforms_getter(dataset, subtask, num_classes, num_clusters):
    if subtask == "rotate":
        def get_transforms(cluster_id, mode="train"):
            angle_step = 360 // num_clusters
            transforms = T.Compose([
                *_get_data_augmentations(dataset, mode),
                T.ToTensor(),
                lambda image: T.functional.rotate(image, angle=angle_step * cluster_id),
                *_get_normalization(dataset),
            ])
            return transforms, lambda y: y

    elif subtask == "label_shift":
        def get_transforms(cluster_id, mode="train"):
            transforms = T.Compose([
                *_get_data_augmentations(dataset, mode),
                T.ToTensor(),
                *_get_normalization(dataset),
            ])

            def label_transforms(y):
                return (y + cluster_id) % num_classes

            return transforms, label_transforms

    else:
        def get_transforms(_, mode="train"):
            transforms = T.Compose([
                *_get_data_augmentations(dataset, mode),
                T.ToTensor(),
                *_get_normalization(dataset),
            ])
            return transforms, lambda y: y
    
    return get_transforms


def get_apply_transforms(transforms, label_transforms, image_key, label_key):
    """Return a function that applies the given transformations to the batch"""
    def apply_transforms(batch):
        batch[image_key] = [transforms(image) for image in batch[image_key]]
        batch[label_key] = [label_transforms(label) for label in batch[label_key]]
        return batch

    return apply_transforms


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
