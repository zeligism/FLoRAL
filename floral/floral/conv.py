
import torch
from torch import nn
import torch.nn.functional as F

from .base import LoRA, Conv
from .utils import resolve_rank, init_lora_

DEFAULT_CONV_OPTS_PER_DIM = {
    kernel_dim: {
        "stride": kernel_dim * (1,),
        "padding": kernel_dim * (0,),
        "dilation": kernel_dim * (1,),
        "groups": 1
    } for kernel_dim in range(3)
}
DEFAULT_METHOD = "balanced_2d"  # seems to be better... what about convtranspose?


class ConvLoRA(LoRA):
    def __init__(self, main: Conv, rank, min_rank=1, bias=False, init_strategy=None, method=None) -> None:
        super().__init__()
        in_kernel_size, out_kernel_size = resolve_kernel_sizes(main, method=method)
        in_conv_opts, out_conv_opts = resolve_conv_opts(main, method=method)
        in_dim = main.in_channels * torch.Tensor(in_kernel_size).numel()
        out_dim = main.out_channels * torch.Tensor(out_kernel_size).numel()
        rank = resolve_rank(rank, in_dim, out_dim, min_rank=min_rank)

        self.kernel_dim = len(main.kernel_size)
        self.layer_in = main.__class__(main.in_channels, rank, in_kernel_size, **in_conv_opts)
        self.layer_out = main.__class__(rank, main.out_channels, out_kernel_size,
                                        bias=bias and main.bias is not None, **out_conv_opts)
        init_lora_(self.layer_in.weight, self.layer_out.weight, in_dim,
                   rank=rank, init_strategy=init_strategy)


class LoraConv2dExperts(nn.Module):
    def __init__(self, main: nn.Conv2d, rank, num_experts, min_rank=1,
                 bias=False, method=None, init_strategy=None, fuse_params=False) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.fuse_params = fuse_params
        self.in_channels = main.in_channels
        self.out_channels = main.out_channels
        self.has_bias = bias and main.bias is not None
        self.in_kernel_size, self.out_kernel_size = resolve_kernel_sizes(main, method=method)
        self.in_conv_opts, self.out_conv_opts = resolve_conv_opts(main, method=method)

        dim_in = self.in_channels * self.in_kernel_size[0] * self.in_kernel_size[1]
        dim_out = self.out_channels * self.out_kernel_size[0] * self.out_kernel_size[1]
        self.rank = resolve_rank(rank, dim_in, dim_out, min_rank=min_rank)

        self.weight_in = nn.Parameter(torch.Tensor(self.num_experts, self.rank, self.in_channels, *self.in_kernel_size))
        self.weight_out = nn.Parameter(torch.Tensor(self.num_experts, self.out_channels, self.rank, *self.out_kernel_size))
        self.bias = nn.Parameter(torch.zeros(num_experts, self.out_channels)) if self.has_bias else None
        init_lora_(self.weight_in, self.weight_out, dim_in, rank=self.rank, init_strategy=init_strategy)

        if self.fuse_params:
            self.fused_weight = nn.Parameter(torch.zeros(num_experts, self.out_channels, self.in_channels, *self.kernel_size))

    @torch.no_grad()
    def fuse(self):
        fused_weight = torch.einsum("cmr...,crn...->cmn...", self.weight_out, self.weight_in)
        if self.fuse_params:
            self.fused_weight.copy_(fused_weight)
        return fused_weight

    @torch.no_grad()
    def defuse(self):
        raise NotImplementedError()

    def forward(self, x, probs):
        outputs = []
        for expert_idx in range(self.num_experts):
            bias = self.bias[expert_idx] if self.has_bias else None
            hidden = F.conv2d(x, weight=self.weight_in[expert_idx], **self.in_conv_opts)
            output = F.conv2d(hidden, weight=self.weight_out[expert_idx], bias=bias, **self.out_conv_opts)
            outputs.append(probs[expert_idx] * output)
        return torch.stack(outputs).sum(0)

        # TODO(optimize): can we make this faster with grouped convs?
        x = x.repeat(1, self.num_experts, *(1,) * (len(x.shape) - 2))
        weight_in = self.weight_in.view(self.num_experts * self.in_channels, self.rank, 1, 1).repeat(1, self.num_experts, 1, 1)
        weight_out = self.weight_out.view(self.num_experts * self.rank, self.out_channels, *self.kernel_size).repeat(1, self.num_experts, 1, 1)
        print(weight_in.shape, weight_out.shape)
        hidden = F.conv2d(x, weight=weight_in, groups=self.num_experts)
        output = F.conv2d(hidden, weight=weight_out, bias=self.bias,
                    stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.num_experts)
        outputs = outputs.view(outputs.shape[0], self.num_experts, self.out_channels, (1,) * (len(outputs.shape) - 3))
        probs = probs.view(1, -1, (1,) * (len(output.shape) - 2))
        output = torch.sum(probs * outputs, dim=1)
        return output


class LoraConv2d(LoraConv2dExperts):
    def __init__(self, main, rank) -> None:
        super().__init__(main, rank, num_experts=1)


def resolve_kernel_sizes(main: nn.Conv2d, method=DEFAULT_METHOD):
    if method is None:
        method = DEFAULT_METHOD
    dim = len(main.kernel_size)
    if method == "balanced_2d" and dim != 2:
        method = "balanced"

    if method == "balanced_2d":
        min_kernel, max_kernel = min(*main.kernel_size), max(*main.kernel_size)
        if main.in_channels > main.out_channels:
            in_kernel_size = (1, min_kernel)
            out_kernel_size = (max_kernel, 1)
        else:
            in_kernel_size = (1, max_kernel)
            out_kernel_size = (min_kernel, 1)
    elif method == "balanced":
        if main.in_channels > main.out_channels:
            in_kernel_size = (1,) * dim
            out_kernel_size = main.kernel_size
        else:
            in_kernel_size = main.kernel_size
            out_kernel_size = (1,) * dim
    elif method == "in":
        in_kernel_size = main.kernel_size
        out_kernel_size = (1,) * dim
    elif method == "out":
        in_kernel_size = (1,) * dim
        out_kernel_size = main.kernel_size
    else:
        raise NotImplementedError(method)

    return in_kernel_size, out_kernel_size


def resolve_conv_opts(main: Conv, method=DEFAULT_METHOD):
    if method is None:
        method = DEFAULT_METHOD
    dim = len(main.kernel_size)
    default_conv_opts = DEFAULT_CONV_OPTS_PER_DIM[dim]
    conv_opts = {"stride": main.stride, "padding": main.padding, "dilation": main.dilation}
    if method == "balanced_2d" and dim != 2:
        method = "balanced"

    if method == "balanced_2d":
        in_conv_opts = {opt: (default_conv_opts[opt][0], conv_opts[opt][1]) for opt in conv_opts.keys()}
        out_conv_opts = {opt: (conv_opts[opt][0], default_conv_opts[opt][1]) for opt in conv_opts.keys()}
    elif method == "out" or (method == "balanced" and main.in_channels > main.out_channels):
        in_conv_opts = {opt: default_conv_opts[opt] for opt in conv_opts.keys()}
        out_conv_opts = {opt: conv_opts[opt] for opt in conv_opts.keys()}
    elif method == "in" or (method == "balanced" and main.in_channels <= main.out_channels):
        in_conv_opts = {opt: conv_opts[opt] for opt in conv_opts.keys()}
        out_conv_opts = {opt: default_conv_opts[opt] for opt in conv_opts.keys()}
    else:
        raise NotImplementedError(method)

    # Ensure no bias in layer_in
    in_conv_opts["bias"] = False

    # XXX: not sure if this will work for groups > 1, need testing
    in_conv_opts["groups"] = main.groups
    out_conv_opts["groups"] = main.groups

    return in_conv_opts, out_conv_opts
