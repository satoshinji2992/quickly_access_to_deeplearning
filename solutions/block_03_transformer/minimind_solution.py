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
    rope_base: int = 10000


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x * rms


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


def repeat_kv(x, repeats):
    if repeats == 1:
        return x
    b, h, t, d = x.shape
    x = x[:, :, None, :, :].expand(b, h, repeats, t, d)
    return x.reshape(b, h * repeats, t, d)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        if config.dim % config.n_heads != 0:
            raise ValueError("dim must be divisible by n_heads")
        if config.n_heads % config.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.dim // config.n_heads
        self.kv_repeats = config.n_heads // config.n_kv_heads
        self.q_proj = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.rope_base = config.rope_base

    def forward(self, x):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = build_rope_cache(t, self.head_dim, self.rope_base, x.device)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        k = repeat_kv(k, self.kv_repeats)
        v = repeat_kv(v, self.kv_repeats)
        scores = q @ k.transpose(-2, -1) / (self.head_dim**0.5)
        mask = torch.tril(torch.ones(t, t, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(b, t, self.n_heads * self.head_dim)
        return self.out_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class DecoderBlock(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.dim)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.dim)
        self.ffn = SwiGLU(config.dim, config.hidden_dim)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None):
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("sequence length exceeds max_seq_len")
        x = self.token_embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.norm(x))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            context = input_ids[:, -self.config.max_seq_len :]
            logits, _ = self(context)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
        return input_ids

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())


def make_toy_batch(batch_size=4, seq_len=16, vocab_size=64, device="cpu"):
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    return x[:, :-1], x[:, 1:]


def smoke_train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = MiniMindConfig(
        vocab_size=64,
        dim=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        hidden_dim=128,
        max_seq_len=32,
    )
    model = MiniMindModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for step in range(5):
        input_ids, labels = make_toy_batch(device=device)
        _, loss = model(input_ids, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"step={step} loss={loss.item():.4f}")
    sample = model.generate(torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=4)
    print("generated:", sample.tolist())


if __name__ == "__main__":
    smoke_train()
