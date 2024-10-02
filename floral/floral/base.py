from typing import Union, Optional
import torch
import torch.nn as nn

Conv = Union[nn.Conv1d, nn.Conv2d, nn.Conv3d]
ConvTranspose = Union[nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d]
BatchNorm = Union[nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]
InstanceNorm = Union[nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d]
Norm = Union[BatchNorm, InstanceNorm, nn.GroupNorm, nn.LayerNorm]
LoRAableModule = Union[nn.Linear, nn.Embedding, Conv, ConvTranspose, Norm]
# Note Conv and ConvTranspose are instances of _ConvND
# Also, BatchNorm and InstanceNorm are instances of _NormBase


class LoRA(nn.Module):
    layer_in: LoRAableModule
    layer_out: LoRAableModule

    @torch.no_grad()
    def reset(self):
        ...

    @staticmethod
    @torch.no_grad()
    def merge(weight_in: torch.Tensor, weight_out: torch.Tensor) -> torch.Tensor:
        ...

    @staticmethod
    @torch.no_grad()
    def demerge(merged_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    def forward(self, *args, **kwargs):
        return self.layer_out(self.layer_in(*args, **kwargs), **kwargs)


class LoRAList(nn.ModuleList):
    _modules: dict[str, LoRA]  # just for type hinting, this is used internally in nn.ModuleList

    @torch.no_grad()
    def reset(self):
        [lora.reset() for lora in self]

    def forward(self, x: torch.Tensor, probs: torch.Tensor, min_prob: float = 0.01):
        probs = probs.view(-1)
        assert len(probs) == len(self)
        probs = probs.clamp(min=min_prob)
        return torch.stack([p * lora(x) for p, lora in zip(probs, self)]).sum(0)
