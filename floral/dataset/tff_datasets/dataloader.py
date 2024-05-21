# https://github.com/facebookresearch/FL_partial_personalization/blob/main/pfl/data/dataloader.py

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""For every dataset in this directory, define:
    - FederatedDataloader
    - ClientDataloader
    - loss_of_batch_fn
    - metrics_of_batch_fn
NOTE: for TFF datasets, use stateless_random operations. 
    Pass the seed using `torch.randint(1<<20, (1,)).item()`.
    We save the PyTorch random seed, so this allows for reproducibility across
    restarted/restored jobs as well.
"""

from typing import Iterator, Iterable


class FederatedDataloader:
    """Pass in a client id and return a dataloader for that client
    """
    def __init__(self, data_dir, client_list, split, batch_size, max_num_elements_per_client):
        pass

    def get_client_dataloader(self, client_id):
        raise NotImplementedError

    def __len__(self):
        """Return number of clients."""
        raise NotImplementedError
    
    def dataset_name(self):
        raise NotImplementedError

    def get_loss_and_metrics_fn(self):
        # loss_fn: return a torch scalar (autodiff enabled)
        # metrics_fn: return an OrderedDict with keys 'loss', 'accuracy', etc.
        raise NotImplementedError

    @property
    def num_classes(self):
        raise NotImplementedError


class ClientDataloader:
    """Dataloader for a client
    """
    def __init__(self):
        pass

    def __iter__(self):
        raise NotImplementedError

    def __next__(self):
        raise NotImplementedError


class TFF_DataLoader(Iterable):
    def __init__(self, dataset: FederatedDataloader, client_id: str) -> None:
        self.dataset = dataset
        self.client_id = client_id
        self.dataloader = []

    def __iter__(self) -> Iterator:
        self.dataloader = self.dataset.get_client_dataloader(self.client_id)
        return self.dataloader

    def __len__(self) -> int:
        return len(self.dataloader)

