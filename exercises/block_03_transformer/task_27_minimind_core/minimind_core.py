"""A small, complete decoder-only language model used throughout Block 3.

The implementation intentionally uses basic PyTorch operations so every tensor
shape remains visible.  It includes the pieces claimed by the tutorial:
RMSNorm, RoPE, causal grouped-query attention, SwiGLU, weight tying, padding
masks and per-layer KV caches.
"""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


KeyValue = tuple[torch.Tensor, torch.Tensor]


@dataclass
class MiniMindConfig:
    vocab_size: int
    dim: int = 256
    n_layers: int = 4
    n_heads: int = 8
    n_kv_heads: int = 4
    hidden_dim: int = 768
    max_seq_len: int = 256
    rope_base: float = 10000.0
    norm_eps: float = 1e-6
    pad_token_id: int | None = 0

    def __post_init__(self):
        positive = {
            "vocab_size": self.vocab_size,
            "dim": self.dim,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "hidden_dim": self.hidden_dim,
            "max_seq_len": self.max_seq_len,
        }
        for name, value in positive.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.dim % self.n_heads != 0:
            raise ValueError("dim must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if (self.dim // self.n_heads) % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        if not math.isfinite(self.rope_base) or self.rope_base <= 0:
            raise ValueError("rope_base must be positive and finite")
        if not math.isfinite(self.norm_eps) or self.norm_eps <= 0:
            raise ValueError("norm_eps must be positive and finite")
        if self.pad_token_id is not None:
            if (
                not isinstance(self.pad_token_id, int)
                or isinstance(self.pad_token_id, bool)
                or not 0 <= self.pad_token_id < self.vocab_size
            ):
                raise ValueError("pad_token_id must be an integer inside the vocabulary")


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
        # Parameters normally remain float32 under mixed precision.  Cast only
        # after the affine scale so RMSNorm preserves the activation dtype.
        return (normalized * self.weight.to(dtype=compute_dtype)).to(dtype=input_dtype)


def build_rope_cache(seq_len, head_dim, base=10000.0, device=None, dtype=None, start_pos=0):
    """Build distinct sin/cos frequencies for each pair of head dimensions."""
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if head_dim <= 0 or head_dim % 2 != 0:
        raise ValueError("head_dim must be a positive even integer")
    if start_pos < 0:
        raise ValueError("start_pos must be non-negative")
    if not math.isfinite(base) or base <= 0:
        raise ValueError("base must be positive and finite")
    pair_index = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
    inv_freq = base ** (-pair_index / head_dim)
    positions = torch.arange(start_pos, start_pos + seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(positions, inv_freq)
    cos, sin = angles.cos(), angles.sin()
    if dtype is not None:
        cos, sin = cos.to(dtype=dtype), sin.to(dtype=dtype)
    return cos, sin


def rotate_half(x):
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((-odd, even), dim=-1).flatten(-2)


def apply_rope(x, cos, sin):
    expected = (x.shape[-2], x.shape[-1] // 2)
    if x.shape[-1] % 2 or cos.shape != expected or sin.shape != expected:
        raise ValueError("RoPE cache shape does not match (seq_len, head_dim // 2)")
    cos = torch.repeat_interleave(cos, 2, dim=-1)[None, None].to(
        device=x.device, dtype=x.dtype
    )
    sin = torch.repeat_interleave(sin, 2, dim=-1)[None, None].to(
        device=x.device, dtype=x.dtype
    )
    return x * cos + rotate_half(x) * sin


def repeat_kv(x, repeats):
    """Map ``n_kv_heads`` K/V heads to all query heads without new parameters."""
    if x.ndim != 4:
        raise ValueError("x must have shape (batch, n_kv_heads, seq_len, head_dim)")
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats <= 0:
        raise ValueError("repeats must be a positive integer")
    if repeats == 1:
        return x
    b, h, t, d = x.shape
    return x[:, :, None].expand(b, h, repeats, t, d).reshape(b, h * repeats, t, d)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.dim // config.n_heads
        self.kv_repeats = config.n_heads // config.n_kv_heads
        self.rope_base = config.rope_base
        self.q_proj = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.dim, config.dim, bias=False)

    def forward(
        self,
        x,
        attention_mask=None,
        past_key_value: KeyValue | None = None,
        use_cache=False,
    ):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)

        past_len = 0
        if past_key_value is not None:
            past_k, past_v = past_key_value
            if past_k.shape[:2] != (b, self.n_kv_heads) or past_v.shape != past_k.shape:
                raise ValueError("past K/V must have shape (batch, n_kv_heads, past_len, head_dim)")
            if past_k.shape[-1] != self.head_dim:
                raise ValueError("past K/V head_dim does not match the model")
            past_len = past_k.shape[-2]

        cos, sin = build_rope_cache(
            t,
            self.head_dim,
            base=self.rope_base,
            device=x.device,
            dtype=x.dtype,
            start_pos=past_len,
        )
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if past_key_value is not None:
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)
        present = (k, v)

        total_len = past_len + t
        q_positions = torch.arange(past_len, total_len, device=x.device)[:, None]
        k_positions = torch.arange(total_len, device=x.device)[None, :]
        allowed = (k_positions <= q_positions)[None, None].expand(b, 1, t, total_len)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=x.device, dtype=torch.bool)
            if attention_mask.shape == (b, t) and past_len:
                prefix = torch.ones((b, past_len), device=x.device, dtype=torch.bool)
                attention_mask = torch.cat((prefix, attention_mask), dim=1)
            if attention_mask.shape != (b, total_len):
                raise ValueError("attention_mask must cover current tokens and cached prefix")
            allowed = allowed & attention_mask[:, None, None, :]

        expanded_k = repeat_kv(k, self.kv_repeats)
        expanded_v = repeat_kv(v, self.kv_repeats)
        scores = q @ expanded_k.transpose(-2, -1) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores.float(), dim=-1).to(dtype=scores.dtype)
        # This also makes an all-padding row finite: its attention output is zero.
        weights = weights * allowed.to(weights.dtype)
        out = weights @ expanded_v
        out = out.transpose(1, 2).contiguous().view(b, t, self.n_heads * self.head_dim)
        out = self.out_proj(out)
        return (out, present) if use_cache else out


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderBlock(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.ffn = SwiGLU(config.dim, config.hidden_dim)

    def forward(self, x, attention_mask=None, past_key_value=None, use_cache=False):
        attn_result = self.attn(
            self.attn_norm(x),
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        if use_cache:
            attn_out, present = attn_result
        else:
            attn_out, present = attn_result, None
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return (x, present) if use_cache else x


class MiniMindCore(nn.Module):
    """A compact but structurally complete decoder-only Transformer."""

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.dim, config.norm_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        # Weight tying must share the Parameter object, not merely equal values.
        self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    @staticmethod
    def _past_length(past_key_values):
        if past_key_values is None:
            return 0
        return past_key_values[0][0].shape[-2]

    def forward(
        self,
        input_ids,
        labels=None,
        attention_mask=None,
        past_key_values: list[KeyValue] | tuple[KeyValue, ...] | None = None,
        use_cache=False,
    ):
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must have shape (batch, non_empty_seq_len)")
        b, t = input_ids.shape
        if past_key_values is not None and len(past_key_values) != len(self.blocks):
            raise ValueError("past_key_values must contain one K/V pair per layer")
        past_len = self._past_length(past_key_values)
        if past_len + t > self.config.max_seq_len:
            raise ValueError("sequence length including cache exceeds max_seq_len")

        if attention_mask is None:
            if past_len:
                attention_mask = torch.ones((b, past_len + t), device=input_ids.device, dtype=torch.bool)
            elif self.config.pad_token_id is None:
                attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
            else:
                attention_mask = input_ids.ne(self.config.pad_token_id)
        else:
            attention_mask = attention_mask.to(device=input_ids.device, dtype=torch.bool)

        x = self.token_embedding(input_ids)
        new_past = []
        for layer_index, block in enumerate(self.blocks):
            layer_past = None if past_key_values is None else past_key_values[layer_index]
            if use_cache:
                x, present = block(
                    x,
                    attention_mask=attention_mask,
                    past_key_value=layer_past,
                    use_cache=True,
                )
                new_past.append(present)
            else:
                x = block(x, attention_mask=attention_mask, past_key_value=layer_past)

        logits = self.lm_head(self.norm(x))
        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must have the same shape as input_ids")
            targets = labels.to(device=input_ids.device).clone()
            if self.config.pad_token_id is not None:
                targets[targets == self.config.pad_token_id] = -100
            query_mask = attention_mask[:, -t:]
            targets[~query_mask] = -100
            valid = targets.ne(-100)
            loss = (
                F.cross_entropy(logits[valid].float(), targets[valid])
                if valid.any()
                else logits.sum() * 0.0
            )
        return (logits, loss, new_past) if use_cache else (logits, loss)

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens,
        temperature=1.0,
        top_k=None,
        top_p=None,
        attention_mask=None,
        eos_token_id=None,
        generator=None,
    ):
        """Reference generation that recomputes the visible context each step.

        Task 30 adds an equivalent cached implementation.  ``temperature=0``
        selects greedy decoding; positive temperatures sample from the logits.
        """
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive")
        if top_p is not None and not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if max_new_tokens == 0:
            return input_ids
        result = input_ids
        if attention_mask is None:
            attention_mask = (
                torch.ones_like(result, dtype=torch.bool)
                if self.config.pad_token_id is None
                else result.ne(self.config.pad_token_id)
            )
        else:
            if attention_mask.shape != result.shape:
                raise ValueError("attention_mask must have the same shape as input_ids")
            attention_mask = attention_mask.to(device=result.device, dtype=torch.bool)
        if not torch.all(attention_mask[:, -1]):
            raise ValueError(
                "each prompt must end in a valid token; left-pad variable-length batches"
            )
        finished = torch.zeros(result.shape[0], device=result.device, dtype=torch.bool)
        for _ in range(max_new_tokens):
            context = result[:, -self.config.max_seq_len :]
            context_mask = attention_mask[:, -self.config.max_seq_len :]
            logits, _ = self(context, attention_mask=context_mask)
            next_logits = logits[:, -1]
            if temperature == 0:
                next_id = next_logits.argmax(dim=-1, keepdim=True)
            else:
                next_logits = next_logits / temperature
                if top_k is not None:
                    k = min(top_k, next_logits.shape[-1])
                    threshold = torch.topk(next_logits, k, dim=-1).values[:, -1:]
                    next_logits = next_logits.masked_fill(next_logits < threshold, float("-inf"))
                if top_p is not None and top_p < 1:
                    sorted_logits, sorted_indices = torch.sort(
                        next_logits, descending=True, dim=-1
                    )
                    cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                    remove_sorted = cumulative > top_p
                    remove_sorted[:, 1:] = remove_sorted[:, :-1].clone()
                    remove_sorted[:, 0] = False
                    remove = torch.zeros_like(remove_sorted).scatter(
                        dim=-1, index=sorted_indices, src=remove_sorted
                    )
                    next_logits = next_logits.masked_fill(remove, float("-inf"))
                probabilities = torch.softmax(next_logits, dim=-1)
                next_id = torch.multinomial(probabilities, 1, generator=generator)
            if eos_token_id is not None:
                next_id = torch.where(
                    finished[:, None],
                    torch.full_like(next_id, eos_token_id),
                    next_id,
                )
                finished |= next_id[:, 0].eq(eos_token_id)
            result = torch.cat((result, next_id), dim=1)
            attention_mask = torch.cat(
                (attention_mask, torch.ones_like(next_id, dtype=torch.bool)), dim=1
            )
            if eos_token_id is not None and torch.all(finished):
                break
        return result

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())


# The reference answer historically used this name; keep it as a readable alias.
MiniMindModel = MiniMindCore


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = MiniMindConfig(vocab_size=32, dim=32, n_layers=2, n_heads=4, n_kv_heads=2, hidden_dim=64)
    model = MiniMindCore(cfg)
    ids = torch.tensor([[1, 5, 8, 2]])
    logits, _ = model(ids)
    print("logits:", tuple(logits.shape))
    print("parameters:", model.parameter_count())
