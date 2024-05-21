from torch import nn
from .base import LoRA


class LayerNormLoRA(LoRA):
    def __init__(self, main: nn.LayerNorm, rank, min_rank=1,
                 bias=False, init_strategy=None) -> None:
        super().__init__()
        main_layer_opts = {
            "eps": main.eps,
            "elementwise_affine": main.elementwise_affine,
            "bias": bias and main.bias is not None,
        }
        self.layer_in = nn.LayerNorm(main.normalized_shape, **main_layer_opts)
        nn.init.zeros_(self.layer_in.weight)
        self.layer_out = nn.Identity()
