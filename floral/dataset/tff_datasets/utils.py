from typing import Optional
from collections import defaultdict
from .dataloader import FederatedDataloader
from .emnist import EmnistDataset
from .shakespeare import ShakespeareDataset
from .stackoverflow import StackOverflowDataset

TFF_DATASETS = ("emnist", "shakespeare", "stackoverflow")


def get_tff_data(dataset,
                 data_dir,
                 statistics_dir="dataset_statistics",
                 client_list=None,
                 batch_size=32,
                 test_batch_size=128,
                 validation_mode=False,
                 **dataset_args,
                 ) -> dict[str, dict[str, Optional[dict[str, FederatedDataloader]]]]:
    assert dataset in TFF_DATASETS
    splits = ("train", "test")
    if validation_mode:
        splits = splits + ("val",)

    if dataset == "emnist":
        TFFDataset = EmnistDataset
    elif dataset == "shakespeare":
        TFFDataset = ShakespeareDataset
    elif dataset == "stackoverflow":
        TFFDataset = StackOverflowDataset
    else:
        raise NotImplementedError(dataset)

    def get_dataset_split(split):
        if split == "train":
            fed_dataset = TFFDataset(
                data_dir, statistics_dir, client_list,
                split="train", batch_size=batch_size, shuffle=True,
                validation_mode=validation_mode, validation_holdout=False, **dataset_args,
            )
        elif split == "val":
            fed_dataset = TFFDataset(
                data_dir, statistics_dir, client_list,
                split="train", batch_size=batch_size, shuffle=False,
                validation_mode=validation_mode, validation_holdout=True, **dataset_args,
            )
        elif split == "test":
            fed_dataset = TFFDataset(
                data_dir, statistics_dir, client_list,
                split="test", batch_size=test_batch_size, shuffle=False,
                validation_mode=validation_mode, validation_holdout=False, **dataset_args,
            )
        else:
            raise NotImplementedError(split)

        return fed_dataset

    dataloaders = defaultdict(dict)
    fed_datasets = {split: get_dataset_split(split) for split in splits}
    for split in splits:
        client_ids_in_split = list(sorted(fed_datasets[split].available_clients_set))
        for client_id in client_ids_in_split:
            # I admit, this is not the best way to do it, but who cares
            if "_tff" not in dataloaders[str(client_id)]:
                dataloaders[str(client_id)]["_tff"] = fed_datasets
            dataloaders[str(client_id)][split] = None

    return dataloaders
