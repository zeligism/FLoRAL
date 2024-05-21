from torch import nn
from math import sqrt, log2
from flwr.common.logger import logger

DEFAULT_INIT_STRATEGY = "uniform_in"


def resolve_rank(rank, dim_in, dim_out, min_rank=1):
    max_rank = min(dim_in, dim_out)
    assert max_rank >= 1
    if rank == "sqrt":
        return round(sqrt(max_rank))
    elif rank in ("log", "log2"):
        return round(log2(1 + max_rank))
    elif rank == "full":
        return max_rank
    else:
        rank = float(rank)
        if rank < 0:
            raise ValueError("Rank cannot be smaller than 0")
        elif rank < 1:
            # e.g. 0.1 means 10% of relative budget (wrt numel of main)
            resolved_rank = int(rank * dim_in * dim_out / (dim_in + dim_out))
            if resolved_rank < min_rank:
                # XXX: This can significantly clutter logs, but is needed at least once.
                #      How to log this warning once? I don't want to pass a variable for this only.
                # logger.warning(f"Couldn't satisfy relative budget of {rank}% "
                #                f"given min rank of {min_rank} "
                #                f"for layer with effective dim: {(dim_in, dim_out)}. "
                #                f"Use min_rank = 0 for strict budget satisfaction.")
                resolved_rank = min_rank
            return resolved_rank
        else:
            return round(rank)


def init_lora_(weight_in, weight_out, fan_in=None,
               rank=1, init_strategy=DEFAULT_INIT_STRATEGY, scale=None):
    assert fan_in is not None or scale is not None
    if init_strategy is None:
        init_strategy = DEFAULT_INIT_STRATEGY
    # kaiming init scales with gain sqrt(2)
    if init_strategy == "uniform_in":
        if scale is None:
            scale = (2 * 3 / fan_in) ** 0.5
        nn.init.uniform_(weight_in, -scale, scale)
        nn.init.zeros_(weight_out)
    elif init_strategy == "uniform_out":
        if scale is None:
            scale = (2 * 3 / rank) ** 0.5
        nn.init.zeros_(weight_in)
        nn.init.uniform_(weight_out, -scale, scale)
    elif init_strategy == "normal_in":
        if scale is None:
            scale = (2 / fan_in) ** 0.5
        nn.init.normal_(weight_in, std=scale)
        nn.init.zeros_(weight_out)
    elif init_strategy == "normal_out":
        if scale is None:
            scale = (2 / rank) ** 0.5
        nn.init.zeros_(weight_in)
        nn.init.normal_(weight_out, std=scale)
    else:
        raise NotImplementedError(init_strategy)

