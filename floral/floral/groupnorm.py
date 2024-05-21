from torch import nn
from .base import LoRA


class GroupNormLoRA(LoRA):
    def __init__(self, main: nn.GroupNorm, rank, min_rank=1,
                 bias=False, init_strategy=None) -> None:
        super().__init__()
        main_layer_opts = {
            "eps": main.eps,
            "affine": main.affine,
        }
        self.layer_in = nn.GroupNorm(main.num_groups, main.num_channels, **main_layer_opts)
        nn.init.zeros_(self.layer_in.weight)
        self.layer_out = nn.Identity()
