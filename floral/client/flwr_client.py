import flwr as fl
from flwr.common.logger import logger
from flwr.common.typing import NDArrays, Scalar, Config
from floral.utils import init_seed, now, get_ndarrays, set_ndarrays
from floral.training import TrainerBase


class FlowerClient(fl.client.NumPyClient):
    def __init__(self,
                 trainer: TrainerBase,
                 save_path: str,
                 client_id: int,
                 ) -> None:
        super().__init__()
        self.trainer = trainer
        self.model = self.trainer.model
        self.save_path = save_path
        self.client_id = client_id

    def get_properties(self, config: Config) -> dict[str, Scalar]:
        properties = {}
        if "has_train" in config:
            del config["has_train"]
            properties["has_train"] = "train" in self.trainer.dataloaders
        
        if "has_test" in config:
            del config["has_test"]
            properties["has_test"] = "test" in self.trainer.dataloaders

        if len(config.keys()) > 0:
            logger.warning(f"Couldn't handle the following get_properties's configurations: {list(config.keys())}")

        return properties

    def get_parameters(self, config: dict[str, Scalar]) -> NDArrays:
        return get_ndarrays(self.model)

    def set_parameters(self, parameters: NDArrays) -> None:
        set_ndarrays(self.model, parameters)

    def fit(self, parameters: NDArrays, config: dict[str, Scalar]) -> tuple[NDArrays, int, dict]:
        round = config["round"]

        try:
            cid_val = int(self.client_id)
        except ValueError:
            cid_val = abs(hash(self.client_id))  # can have dups, but this is good enough

        try:
            data_len = len(self.trainer.dataloaders['train'].dataset)
        except TypeError:
            # for tff, length is precomputed (__len__ can raise unknown length error)
            data_len = self.trainer.dataloaders['train'].dataset_size

        init_seed([round, cid_val])
        start_time = now()
        self.set_parameters(parameters)
        train_metrics = self.trainer.train()
        metrics = {
            "round": round,
            "duration": now() - start_time,
            **train_metrics,
        }

        return (
            self.get_parameters(self.model),
            data_len,
            metrics,
        )

    def evaluate(self, parameters: NDArrays, config: dict[str, Scalar]):
        """Evaluate the model sent by the server on this client's
        local validation set. Then return performance metrics."""
        if 'test' not in self.trainer.dataloaders:
            logger.warning(f"Client {self.client_id} does not have a test data loader for evaluation")
            return 0., 0, {}

        try:
            data_len = len(self.trainer.dataloaders['test'].dataset)
        except TypeError:
            # for tff, length is precomputed (__len__ can raise unknown length error)
            data_len = self.trainer.dataloaders['test'].dataset_size

        start_time = now()
        self.set_parameters(parameters)
        eval_metrics = self.trainer.evaluate()
        metrics = {
            "duration": now() - start_time,
            **eval_metrics,
        }

        return (
            eval_metrics["loss"],
            data_len,
            metrics,
        )

