import torch
from torch import nn


class TransformerBlock(nn.Module):
    def __init__(self, dim, attention, feed_forward):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = attention
        self.norm2 = nn.LayerNorm(dim)
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

