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

    def reset(self):
        ...

    def fuse(self):
        ...

    def defuse(self):
        ...

    def forward(self, *args, **kwargs):
        return self.layer_out(self.layer_in(*args, **kwargs), **kwargs)


class LoRAList(nn.ModuleList):
    _modules: dict[str, LoRA]

    @torch.no_grad()
    def reset(self):
        [lora.reset() for lora in self]

    @torch.no_grad()
    def fuse(self):
        return [lora.fuse() for lora in self]

    @torch.no_grad()
    def defuse(self):
        return [lora.defuse() for lora in self]
    
    def forward(self, x: torch.Tensor, probs: Optional[torch.Tensor]):
        probs = probs.view(-1)
        assert len(probs) == len(self)
        lora_outputs = torch.stack([lora(x) for lora in self])
        return torch.einsum("c,c...->...", probs, lora_outputs)
