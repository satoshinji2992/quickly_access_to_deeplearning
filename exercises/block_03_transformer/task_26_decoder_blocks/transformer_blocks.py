import math

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        if dim <= 0 or not math.isfinite(eps) or eps <= 0:
            raise ValueError("dim and eps must be positive, with eps finite")
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        input_dtype = x.dtype
        compute_dtype = (
            torch.float32
            if input_dtype in (torch.float16, torch.bfloat16)
            else input_dtype
        )
        x_compute = x.to(dtype=compute_dtype)
        normalized = x_compute * torch.rsqrt(
            x_compute.square().mean(dim=-1, keepdim=True) + self.eps
        )
        return (normalized * self.weight.to(dtype=compute_dtype)).to(dtype=input_dtype)


class TransformerBlock(nn.Module):
    def __init__(self, dim, attention, feed_forward):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attention = attention
        self.norm2 = RMSNorm(dim)
        self.feed_forward = feed_forward

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x


class TransformerStack(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


if __name__ == "__main__":
    blocks = [
        TransformerBlock(
            16,
            attention=nn.Linear(16, 16, bias=False),
            feed_forward=nn.Sequential(nn.Linear(16, 32), nn.SiLU(), nn.Linear(32, 16)),
        )
        for _ in range(2)
    ]
    values = torch.randn(2, 5, 16)
    print("stack output:", tuple(TransformerStack(blocks)(values).shape))
    print("normalization: RMSNorm")
