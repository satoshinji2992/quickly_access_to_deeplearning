import math

import torch
from torch import nn


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def _split_heads(self, x):
        b, t, d = x.shape
        x = x.view(b, t, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def forward(self, query, memory, mask=None):
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(memory))
        v = self._split_heads(self.v_proj(memory))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(query.shape[0], query.shape[1], self.dim)
        return self.out_proj(out), attn

