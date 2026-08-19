import math

import torch


def build_rope_cache(
    seq_len,
    head_dim,
    base=10000,
    device=None,
    dtype=None,
    start_pos=0,
):
    """Return one cosine/sine value for every two-dimensional RoPE plane.

    Column ``i`` uses angular frequency ``base ** (-2 * i / head_dim)``;
    adjacent two-dimensional planes therefore do *not* share one frequency.
    ``start_pos`` is needed when decoding after a KV cache prefix.
    """
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if head_dim <= 0 or head_dim % 2 != 0:
        raise ValueError("head_dim must be a positive even integer for RoPE")
    if start_pos < 0:
        raise ValueError("start_pos must be non-negative")
    if not math.isfinite(base) or base <= 0:
        raise ValueError("base must be positive and finite")
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    positions = torch.arange(start_pos, start_pos + seq_len, device=device).float()
    freqs = torch.outer(positions, inv_freq)
    cos, sin = torch.cos(freqs), torch.sin(freqs)
    if dtype is not None:
        cos, sin = cos.to(dtype=dtype), sin.to(dtype=dtype)
    return cos, sin


def rotate_half(x):
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rope(x, cos, sin):
    if x.shape[-1] % 2 != 0:
        raise ValueError("the last dimension must be even for RoPE")
    if cos.shape != sin.shape or cos.shape != (x.shape[-2], x.shape[-1] // 2):
        raise ValueError("cos/sin must have shape (seq_len, head_dim // 2)")
    cos = torch.repeat_interleave(cos, repeats=2, dim=-1)[None, None, :, :]
    sin = torch.repeat_interleave(sin, repeats=2, dim=-1)[None, None, :, :]
    cos = cos.to(device=x.device, dtype=x.dtype)
    sin = sin.to(device=x.device, dtype=x.dtype)
    return x * cos + rotate_half(x) * sin


if __name__ == "__main__":
    values = torch.randn(2, 4, 6, 8)
    rope_cos, rope_sin = build_rope_cache(6, 8)
    rotated = apply_rope(values, rope_cos, rope_sin)
    print("input/output:", tuple(values.shape), tuple(rotated.shape))
    print("pair frequencies differ:", not torch.allclose(rope_cos[1, 0], rope_cos[1, 1]))
