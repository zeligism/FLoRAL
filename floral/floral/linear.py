
import torch
from torch import nn
import torch.nn.functional as F
from .base import LoRA
from .utils import resolve_rank, init_lora_


class LinearLoRA(LoRA):
    def __init__(self, main: nn.Linear, rank, min_rank=1, bias=False, init_strategy=None) -> None:
        super().__init__()
        rank = resolve_rank(rank, main.in_features, main.out_features, min_rank=min_rank)
        self.layer_in = nn.Linear(main.in_features, rank, bias=False)  # no bias in layer_in
        self.layer_out = nn.Linear(rank, main.out_features, bias=bias and main.bias is not None)
        init_lora_(self.layer_in.weight, self.layer_out.weight, main.in_features,
                   rank=rank, init_strategy=init_strategy)


class LoraLinearExperts(nn.Module):
    def __init__(self, main: nn.Linear, rank, num_experts, min_rank=1,
                 bias=False, init_strategy=None, fuse_params=False) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.fuse_params = fuse_params
        self.in_features = main.in_features
        self.out_features = main.out_features
        self.has_bias = bias and main.bias is not None
        self.rank = resolve_rank(rank, self.in_features, self.out_features, min_rank=min_rank)

        self.weight_in = nn.Parameter(torch.Tensor(num_experts, self.rank, self.in_features))
        self.weight_out = nn.Parameter(torch.Tensor(num_experts, self.out_features, self.rank))
        self.bias = nn.Parameter(torch.zeros(num_experts, self.out_features)) if self.has_bias else None
        init_lora_(self.weight_in, self.weight_out, self.in_features, rank=self.rank, init_strategy=init_strategy)

        if self.fuse_params:
            self.fused_weight = nn.Parameter(torch.einsum("cmr,crn->cmn", self.weight_out, self.weight_in))

    def fuse(self):
        fused_weight = torch.einsum("cmr,crn->cmn", self.weight_out, self.weight_in)
        if self.fuse_params:
            self.fused_weight.copy_(fused_weight)
        return fused_weight

    # TODO(experimental): defuse
    @torch.no_grad()
    def defuse(self, s_pow=1.):
        assert self.fuse_params
        # mean_loras = self.fused_weight.mean(0)
        # mean_loras = torch.einsum("c,cmn->mn", probs, self.fused_weight)
        # self.main_weight.add_(mean_loras)
        # self.fused_weight.sub_(mean_loras.unsqueeze(0))
        for i in range(self.num_experts):
            U, S, Vt = torch.linalg.svd(self.fused_weight[i])
            U, S, Vt = U[:, :self.rank], S[:self.rank], Vt[:self.rank, :]
            # weight_in_i = torch.diag(S**s_pow) @ Vt
            # weight_out_i = U @ torch.diag(S**(1-s_pow))
            # weight_in_i = torch.einsum("ki,kj,rj->ri", Vt, Vt, self.weight_in[i])  # (self.weight_in[i] @ Vt.T) @ Vt
            # weight_out_i = torch.einsum("ik,jk,jr->ir", U, U, self.weight_out[i])  # U @ (U.T @ self.weight_out[i])
            weight_in_i = U @ ((self.weight_in[i].T @ self.weight_in[i]) @ S.pow(-0.5))
            weight_out_i = (S.pow(-0.5) @ (self.weight_out[i] @ self.weight_out[i].T)) @ Vt
            self.weight_in[i].copy_(weight_in_i)
            self.weight_out[i].copy_(weight_out_i)

    def forward(self, x, probs):
        p_BAx = torch.einsum("c,cmr,crn,...n->...m", probs, self.weight_out, self.weight_in, x)
        if self.has_bias:
            p_bias = torch.einsum("c,cm->m", probs, self.bias)
            p_bias = p_bias.view(*(1,) * (len(p_BAx.shape) - 1), -1)
            return p_BAx + p_bias
        else:
            return p_BAx
