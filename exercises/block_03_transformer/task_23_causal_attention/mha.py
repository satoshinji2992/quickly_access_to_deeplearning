import math

import torch
from torch import nn


class MultiHeadSelfAttention(nn.Module):
    """Causal self-attention with RoPE and grouped-query attention (GQA)."""

    def __init__(self, dim, num_heads, num_kv_heads=None, rope_base=10000):
        super().__init__()
        num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        if num_heads <= 0 or num_kv_heads <= 0:
            raise ValueError("num_heads and num_kv_heads must be positive")
        if not math.isfinite(rope_base) or rope_base <= 0:
            raise ValueError("rope_base must be positive and finite")
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        self.kv_repeats = num_heads // num_kv_heads
        self.rope_base = rope_base
        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    @staticmethod
    def _rope(x, start_pos=0, base=10000):
        half = torch.arange(0, x.shape[-1], 2, device=x.device).float()
        inv_freq = base ** (-half / x.shape[-1])
        positions = torch.arange(
            start_pos, start_pos + x.shape[-2], device=x.device
        ).float()
        angles = torch.outer(positions, inv_freq)
        cos = torch.repeat_interleave(angles.cos(), 2, dim=-1).to(x.dtype)
        sin = torch.repeat_interleave(angles.sin(), 2, dim=-1).to(x.dtype)
        even, odd = x[..., 0::2], x[..., 1::2]
        rotated = torch.stack((-odd, even), dim=-1).flatten(-2)
        return x * cos[None, None] + rotated * sin[None, None]

    def _repeat_kv(self, x):
        if self.kv_repeats == 1:
            return x
        b, h, t, d = x.shape
        return (
            x[:, :, None]
            .expand(b, h, self.kv_repeats, t, d)
            .reshape(b, self.num_heads, t, d)
        )

    def forward(self, x, causal=True, attention_mask=None):
        b, t, d = x.shape
        q = self.q_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = self._rope(q, base=self.rope_base)
        k = self._repeat_kv(self._rope(k, base=self.rope_base))
        v = self._repeat_kv(v)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        allowed = torch.ones(t, t, device=x.device, dtype=torch.bool)
        if causal:
            allowed = torch.tril(allowed)
        allowed = allowed[None, None].expand(b, 1, t, t)
        if attention_mask is not None:
            if attention_mask.shape != (b, t):
                raise ValueError("attention_mask must have shape (batch, seq_len)")
            mask = attention_mask.to(device=x.device, dtype=torch.bool)
            allowed = allowed & mask[:, None, None, :]
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        attn = attn * allowed.to(attn.dtype)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(b, t, d)
        return self.out_proj(out)


if __name__ == "__main__":
    layer = MultiHeadSelfAttention(dim=32, num_heads=4, num_kv_heads=2)
    values = torch.randn(2, 6, 32)
    print("output:", tuple(layer(values).shape))
    print("Q heads / KV heads:", layer.num_heads, "/", layer.num_kv_heads)
