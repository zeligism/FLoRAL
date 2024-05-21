# Inspired by flwr_datasets.partitioner.NaturalIdPartitioner

import datasets
from flwr_datasets.partitioner.partitioner import Partitioner
from flwr.common.logger import logger


class NaturalIdClusterPartitioner(Partitioner):
    def __init__(
        self,
        num_clusters: int,
        num_nodes_per_cluster: int,
        partition_by: str,
    ):
        super().__init__()
        self._num_clusters = num_clusters
        self._num_nodes_per_cluster = num_nodes_per_cluster
        self._partition_by = partition_by
        # partition the natural ids in the clusters (where cluster_id = node_id % self._num_clusters)
        self._cluster_id_to_natural_ids = {cluster_id: set() for cluster_id in range(self._num_clusters)}
        # lazy initialization
        self._initialized = False

    def _create_int_cluster_id_to_natural_ids(self) -> None:
        unique_natural_ids = self.dataset.unique(self._partition_by)
        if len(unique_natural_ids) % self._num_clusters > 0:
            logger.warning("Labels cannot be equally assigned to clusters. "
                           "Some clusters will have more labels than others")
        try:
            for natural_id in unique_natural_ids:
                i = int(natural_id)
                self._cluster_id_to_natural_ids[i % self._num_clusters].add(natural_id)
        except ValueError:
            for i, natural_id in enumerate(sorted(unique_natural_ids)):
                self._cluster_id_to_natural_ids[i % self._num_clusters].add(natural_id)

    def load_partition(self, node_id: int) -> datasets.Dataset:
        assert node_id <= self._num_clusters * self._num_nodes_per_cluster, "Unexpected node id"

        if not self._initialized:
            self._create_int_cluster_id_to_natural_ids()
            self._initialized = True

        cluster_id = node_id % self._num_clusters
        node_id_within_cluster = node_id // self._num_clusters
        cluster_dataset = self.dataset.filter(
            lambda row: row[self._partition_by] in self._cluster_id_to_natural_ids[cluster_id]
        )
        node_dataset = cluster_dataset.shard(
            num_shards=self._num_nodes_per_cluster, index=node_id_within_cluster, contiguous=True
        )
        return node_dataset

    def num_partitions(self) -> int:
        return self._num_clusters * self._num_nodes_per_cluster

    @property
    def cluster_id_to_natural_ids(self) -> dict[int, str]:
        return self._cluster_id_to_natural_ids

    @cluster_id_to_natural_ids.setter
    def cluster_id_to_natural_ids(self, value: dict[int, str]) -> None:
        raise AttributeError(
            "Setting the _cluster_id_to_natural_ids dictionary is not allowed."
        )
