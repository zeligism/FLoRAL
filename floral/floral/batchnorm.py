
import torch
from torch import nn
import torch.nn.functional as F
from .base import LoRA, BatchNorm
from .utils import resolve_rank
import torch.nn.utils.parametrize as parametrize


class BatchNormLoRA(LoRA):
    def __init__(self, main: BatchNorm, rank, min_rank=1,
                 bias=False, init_strategy=None, reparam=False) -> None:
        super().__init__()
        in_dim = main.num_features
        out_dim = main.num_features
        rank = resolve_rank(rank, in_dim, out_dim, min_rank=min_rank)

        # Use exactly the same original batchnorm arguments in the in layer
        bn_opts = {
            "eps": main.eps,
            "momentum": main.momentum,
            "affine": main.affine,
            "track_running_stats": main.track_running_stats,
        }
        self.layer_in = main.__class__(main.num_features, **bn_opts)
        # But with weight initialized to 0 (thus, if affine=False, this lora will always output 0)
        nn.init.zeros_(self.layer_in.weight)
        nn.init.zeros_(self.layer_in.bias)  # it's already zero, but just in case
        # Out layer is a dummy (another impl can be to simulate the zero init above in this layer)
        self.layer_out = nn.Identity()

        # XXX: experimental idea: for when this batchnorm is kept private at the client.
        # Adding BatchNormLoRA's reparameterized params to the main BatchNorm's params is made
        # invariant to the local BatchNormLoRA's running stats.
        # In other words, we have adaptation under normalization wrt main running stats:
        #   main_bn(x) + reparam_lora_bn(x) = main_bn(x, main_weight + reparam_lora_weight, main_bias + reparam_lora_bias)
        if reparam:
            parametrize.register_parametrization(self.layer_in, "weight", BatchNormWeightReparam(main, self.layer_in))
            parametrize.register_parametrization(self.layer_in, "bias", BatchNormBiasReparam(main, self.layer_in))


class BatchNormReparam(nn.Module):
    def __init__(self, main, lora) -> None:
        super().__init__()
        # hide inside a tuple to avoid registration
        self._main_bn = (main,)
        self._lora_bn = (lora,)

    @property
    def main_bn(self) -> BatchNorm:
        return self._main_bn[0]

    @property
    def lora_bn(self) -> BatchNorm:
        return self._lora_bn[0]


class BatchNormWeightReparam(BatchNormReparam):
    def forward(self, weight: torch.Tensor):
        rel_dev = self.lora_bn.running_var.sqrt().div(self.main_bn.running_var.sqrt().add(self.lora_bn.eps))
        reparameterized_weight = weight.mul(rel_dev)
        return reparameterized_weight


class BatchNormBiasReparam(BatchNormReparam):
    def forward(self, bias: torch.Tensor):
        shift = self.lora_bn.running_mean.sub(self.main_bn.running_mean)
        normalized_shift = shift.div(self.main_bn.running_var.sqrt().add(self.lora_bn.eps))
        reparameterized_bias = bias.add(self.lora_bn.weight.detach().mul(normalized_shift))
        return reparameterized_bias



class LoraBatchNormExperts(nn.Module):
    def __init__(self, main, _, num_experts=1, bias=False, init_strategy=None, fuse_params=False) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.fuse_params = fuse_params
        self.has_bias = bias and main.bias is not None
        if isinstance(main, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            self.norm_type = "BatchNorm"
            self.num_features = main.num_features
            # save pointers to main running stats
            self.main_running_mean = main.running_mean
            self.main_running_var = main.running_var
        elif isinstance(main, nn.GroupNorm):
            self.norm_type = "GroupNorm"
            self.num_features = main.num_channels
        elif isinstance(main, nn.LayerNorm):
            self.norm_type = "LayerNorm"
            self.num_features = main.num_channels
        else:
            raise NotImplementedError(f"Norm of type '{type(main)}' is not implemented.")
        self.eps = main.eps

        self.weight_in = nn.Parameter(torch.zeros(num_experts, 1))
        self.weight_out = nn.Parameter(torch.ones(num_experts, self.num_features))
        self.bias = nn.Parameter(torch.zeros(num_experts, self.num_features))
        if self.fuse_params:
            self.fused_weight = nn.Parameter(torch.zeros(num_experts, self.num_features))

    def forward(self, x, probs, reparam=True):
        # XXX: reparam batchnorm? (Have to have local running stats)
        if reparam and self.norm_type == "BatchNorm":
            x = x.sub(self.main_running_mean).div(self.main_running_var.sqrt().add(self.eps))

        xs = x.repeat(1, self.num_experts, *(1,) * (len(x.shape) - 2))  # channel dim =  e groups of x (e x d)
        normalized_xs = F.group_norm(xs, self.num_experts, eps=self.eps)  # normalize per expert without affine
        normalized_xs = torch.stack(normalized_xs.split(self.num_features, dim=1), dim=1)  # split channel dim into e,d again
        # view non-channel-wise dims as 1 and affine transform x channel-wise
        weight = self.weight_out * self.weight_in
        weight = weight.view(1, *weight.size(), *(1,) * (len(normalized_xs.shape) - 3))
        bias = self.bias.view(1, *self.bias.size(), *(1,) * (len(normalized_xs.shape) - 3))
        outputs = weight * normalized_xs + bias
        # weight experts according to prob and expert-wise sum
        probs = probs.view(1, -1, *(1,) * (len(normalized_xs.shape) - 2))
        output = torch.sum(probs * outputs, dim=1)
        return output

    # def functional_forward(self, x, bn, weight, bias):
    #     return F.batch_norm(x, running_mean=bn.running_mean,
    #                         running_var=bn.running_var,
    #                         weight=weight, bias=bias, training=bn.training,
    #                         momentum=bn.momentum, eps=bn.eps)

    # def forward(self, x, probs, reparam=True):
    #     outputs = []
    #     for i in range(probs.size(0)):
    #         bn = self.bn_out[i]
    #         weight = self.weight_in[i] * bn.weight
    #         bias = bn.bias
    #         if reparam:
    #             # bias
    #             shift = bn.running_mean.sub(self.main_running_mean)
    #             normalized_shift = shift.div(self.main_running_var.sqrt().add(bn.eps))
    #             bias = bias.add(weight.detach().mul(normalized_shift))
    #             # weight
    #             rel_dev = bn.running_var.sqrt().div(self.main_running_var.sqrt().add(bn.eps))
    #             weight = weight.mul(rel_dev)
    #         out = self.functional_forward(x, bn, weight, bias)
    #         outputs.append(probs[i] * out)
    #     return torch.stack(outputs, dim=0).sum(0)

