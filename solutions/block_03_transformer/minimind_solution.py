"""Block 3 reference entry point.

The exercise core is complete, so the solution reuses it instead of keeping a
second implementation that can drift. Its ``generate`` method uses task 30's
real per-layer KV cache.
"""

from pathlib import Path
import sys

import torch


BLOCK = Path(__file__).resolve().parents[2] / "exercises" / "block_03_transformer"
CORE = BLOCK / "task_27_minimind_core"
CACHE = BLOCK / "task_30_kv_cache"
for folder in (CORE, CACHE):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from minimind_core import (  # noqa: E402,F401
    CausalSelfAttention,
    DecoderBlock,
    MiniMindConfig,
    MiniMindCore,
    RMSNorm,
    SwiGLU,
    apply_rope,
    build_rope_cache,
    repeat_kv,
    rotate_half,
)
from kv_cache import cache_equivalence_error, generate_with_kv_cache  # noqa: E402


class MiniMindModel(MiniMindCore):
    """Reference model with cached generation enabled by default."""

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
        return generate_with_kv_cache(
            self,
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            attention_mask=attention_mask,
            eos_token_id=eos_token_id,
            generator=generator,
        )


def smoke_test():
    torch.manual_seed(0)
    config = MiniMindConfig(
        vocab_size=64,
        dim=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        hidden_dim=128,
        max_seq_len=32,
        pad_token_id=0,
    )
    model = MiniMindModel(config).eval()
    input_ids = torch.tensor([[1, 7, 11, 9, 2]])
    logits, _ = model(input_ids)
    generated = model.generate(input_ids, max_new_tokens=4, temperature=0)
    print("logits:", tuple(logits.shape))
    print("cached/full max_abs_error:", f"{cache_equivalence_error(model, input_ids):.3e}")
    print("generated:", generated.tolist())


if __name__ == "__main__":
    smoke_test()
