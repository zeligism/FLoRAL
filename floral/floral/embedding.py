
import torch
from torch import nn
import torch.nn.functional as F

from .base import LoRA
from .utils import resolve_rank, init_lora_


class EmbeddingLoRA(LoRA):
    def __init__(self, main: nn.Embedding, rank, min_rank=1, bias=False, init_strategy=None) -> None:
        super().__init__()
        rank = resolve_rank(rank, main.num_embeddings, main.embedding_dim, min_rank=min_rank)
        self.layer_in = nn.Embedding(main.num_embeddings, rank)
        self.layer_out = nn.Linear(rank, main.embedding_dim, bias=False)  # no bias in embedding
        init_lora_(self.layer_in.weight, self.layer_out.weight, init_strategy="normal_in", scale=1.0)


class LoraEmbeddingExperts(nn.Module):
    def __init__(self, main: nn.Embedding, rank, num_experts, min_rank=1, bias=False, init_strategy=None, fuse_params=False) -> None:
        super().__init__()
        self.num_loras = num_experts
        self.num_embeddings = main.num_embeddings
        self.embedding_dim = main.embedding_dim
        self.has_bias = False
        self.rank = resolve_rank(rank, self.num_embeddings, self.embedding_dim, min_rank=min_rank)

        self.layer_in = nn.ModuleList([
            nn.Embedding(self.num_embeddings, self.rank) for _ in range(self.num_loras)
        ])
        self.layer_out = nn.ModuleList([
            nn.Linear(self.rank, self.embedding_dim, bias=False) for _ in range(self.num_loras)
        ])
        self.bias = None
        for w_in, w_out in zip(self.layer_in, self.layer_out):
            init_lora_(w_in.weight, w_out.weight, self.num_embeddings,
                    rank=self.rank, init_strategy="normal_in", scale=1.0)

    def fuse(self):
        raise NotImplementedError

    def forward(self, x, probs):
        outputs = torch.stack([
            self.layer_out[c](self.layer_in[c](x)) for c in range(self.num_loras)
        ])
        weighted_output = torch.einsum("c,c...", probs, outputs)
        return weighted_output
