from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class MiniMindConfig:
    vocab_size: int
    dim: int = 256
    n_layers: int = 4
    n_heads: int = 8
    n_kv_heads: int = 4
    hidden_dim: int = 768
    max_seq_len: int = 256


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x * rms


class MiniMindCore(nn.Module):
    """Teaching skeleton aligned with MiniMind-style decoder-only models.

    TODO in later tasks: replace the placeholder body with RoPE + GQA blocks.
    """

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    RMSNorm(config.dim),
                    nn.Linear(config.dim, config.hidden_dim),
                    nn.SiLU(),
                    nn.Linear(config.hidden_dim, config.dim),
                )
                for _ in range(config.n_layers)
            ]
        )
        self.norm = RMSNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

    def forward(self, input_ids, labels=None):
        x = self.token_embedding(input_ids)
        for block in self.blocks:
            x = x + block(x)
        logits = self.lm_head(self.norm(x))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        return logits, loss

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

