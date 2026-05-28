import torch


def build_rope_cache(seq_len, head_dim, base=10000, device=None):
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for RoPE")
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(positions, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def rotate_half(x):
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rope(x, cos, sin):
    cos = torch.repeat_interleave(cos, repeats=2, dim=-1)[None, None, :, :]
    sin = torch.repeat_interleave(sin, repeats=2, dim=-1)[None, None, :, :]
    return x * cos + rotate_half(x) * sin

