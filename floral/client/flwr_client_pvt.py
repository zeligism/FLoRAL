
import os
import torch
from .flwr_client import FlowerClient
from flwr.common.logger import logger
from flwr.common.typing import NDArrays, Scalar
from floral.utils import dict_to_ndarrays, ndarrays_to_dict, split_state_dict


class FlowerClientWithPrivateModules(FlowerClient):
    def __init__(self, *args,
                 private_dir: str,
                 private_modules: set[str] = None,
                 full_comm: bool = True,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.private_dir = private_dir
        self.full_comm = full_comm
        self.private_dict_path = os.path.join(
            self.save_path, self.private_dir, f"{self.client_id}.pt")
        self.set_private_modules(private_modules)
        # to enable control of persistent states saving / loading

    def set_private_modules(self, private_modules: list[str]):
        if private_modules is None:
            private_modules = set()
        self._private_modules = set(private_modules)

    def add_private_modules(self, private_modules: list[str]):
        self._private_modules |= set(private_modules)

    def remove_private_modules(self, private_modules: list[str]):
        self._private_modules -= set(private_modules)

    def get_parameters(self, config: dict[str, Scalar]) -> NDArrays:
        # Get all dict partitions and specify outgoing dict
        state_dict = self.model.state_dict()
        public_dict, private_dict = split_state_dict(state_dict, self._private_modules)
        outgoing_dict = state_dict if self.full_comm else public_dict
        # Save private dict and send the `outgoing_dict`
        if len(private_dict) > 0:
            logger.debug(f"Client {self.client_id}: saving private_dict to '{self.private_dict_path}'")
            torch.save(private_dict, self.private_dict_path)
        parameters = dict_to_ndarrays(outgoing_dict)
        return parameters

    def set_parameters(self, parameters: NDArrays) -> None:
        # Get all dict partitions and specify incoming dict
        state_dict = self.model.state_dict()
        public_dict, private_dict = split_state_dict(state_dict, self._private_modules)
        incoming_dict = ndarrays_to_dict(
            state_dict.keys() if self.full_comm else public_dict.keys(),
            parameters
        )
        # If private dict file exists, use it. Otherwise,
        # use the default private dict above (e.g. in 1st round).
        if os.path.exists(self.private_dict_path):
            logger.debug(f"Client {self.client_id}: loading private_dict from '{self.private_dict_path}'")
            private_dict = torch.load(self.private_dict_path)
        # Add or override private dict
        incoming_dict.update(private_dict)
        self.model.load_state_dict(incoming_dict, strict=True)


class LocalClient(FlowerClientWithPrivateModules):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.set_private_modules([""])  # an empty string matches with any string

