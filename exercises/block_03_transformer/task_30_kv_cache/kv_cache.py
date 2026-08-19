"""Per-layer KV-cache decoding and an executable full-vs-cached check."""

import argparse
from pathlib import Path
import sys

import torch


BLOCK = Path(__file__).resolve().parents[1]
TRAINING = BLOCK / "task_28_next_token_training"
SAMPLING = BLOCK / "task_29_generate_sampling"
for folder in (TRAINING, SAMPLING):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

def sample_next_token(
    logits, temperature=1.0, top_k=None, top_p=None, generator=None
):
    """Local copy of task 29's tiny sampling primitive for import safety."""
    if logits.ndim != 2:
        raise ValueError("logits must have shape (batch, vocab_size)")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive")
    if top_p is not None and not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    filtered = logits / temperature
    if top_k is not None:
        k = min(top_k, filtered.shape[-1])
        threshold = torch.topk(filtered, k, dim=-1).values[:, -1:]
        filtered = filtered.masked_fill(filtered < threshold, float("-inf"))
    if top_p is not None and top_p < 1:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove_sorted = cumulative > top_p
        remove_sorted[:, 1:] = remove_sorted[:, :-1].clone()
        remove_sorted[:, 0] = False
        remove = torch.zeros_like(remove_sorted).scatter(
            dim=-1, index=sorted_indices, src=remove_sorted
        )
        filtered = filtered.masked_fill(remove, float("-inf"))
    return torch.multinomial(torch.softmax(filtered, dim=-1), 1, generator=generator)


def _default_mask(model, input_ids):
    pad_id = model.config.pad_token_id
    return torch.ones_like(input_ids, dtype=torch.bool) if pad_id is None else input_ids.ne(pad_id)


@torch.no_grad()
def prefill(model, input_ids, attention_mask=None):
    """Process a prompt once and return its logits plus one K/V pair per layer."""
    if attention_mask is None:
        attention_mask = _default_mask(model, input_ids)
    logits, _, cache = model(input_ids, attention_mask=attention_mask, use_cache=True)
    return logits, cache


@torch.no_grad()
def decode_one(model, input_id, past_key_values, attention_mask=None):
    """Decode one new input token while reusing all prefix K/V tensors."""
    if input_id.ndim != 2 or input_id.shape[1] != 1:
        raise ValueError("input_id must have shape (batch, 1)")
    past_len = past_key_values[0][0].shape[-2]
    if attention_mask is None:
        attention_mask = torch.ones(
            (input_id.shape[0], past_len + 1), device=input_id.device, dtype=torch.bool
        )
    logits, _, cache = model(
        input_id,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        use_cache=True,
    )
    return logits, cache


@torch.no_grad()
def logits_with_kv_cache(model, input_ids, attention_mask=None):
    """Return every position's logits by prefill(1) + repeated cached decoding."""
    if input_ids.shape[1] > model.config.max_seq_len:
        raise ValueError("input is longer than max_seq_len")
    if attention_mask is None:
        attention_mask = _default_mask(model, input_ids)
    first_logits, cache = prefill(model, input_ids[:, :1], attention_mask[:, :1])
    pieces = [first_logits]
    for position in range(1, input_ids.shape[1]):
        step_logits, cache = decode_one(
            model,
            input_ids[:, position : position + 1],
            cache,
            attention_mask=attention_mask[:, : position + 1],
        )
        pieces.append(step_logits)
    return torch.cat(pieces, dim=1)


@torch.no_grad()
def cache_equivalence_error(model, input_ids, attention_mask=None):
    """Maximum absolute error between ordinary causal forward and cached forward."""
    if attention_mask is None:
        attention_mask = _default_mask(model, input_ids)
    full_logits, _ = model(input_ids, attention_mask=attention_mask)
    cached_logits = logits_with_kv_cache(model, input_ids, attention_mask)
    return (full_logits - cached_logits).abs().max().item()


@torch.no_grad()
def generate_with_kv_cache(
    model,
    input_ids,
    max_new_tokens,
    temperature=1.0,
    top_k=None,
    top_p=None,
    attention_mask=None,
    eos_token_id=None,
    generator=None,
):
    """Generate with real per-layer caches, rebuilding only after window rollover."""
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
        attention_mask = _default_mask(model, result)
    elif attention_mask.shape != result.shape:
        raise ValueError("attention_mask must have the same shape as input_ids")
    else:
        attention_mask = attention_mask.to(device=result.device, dtype=torch.bool)
    if not torch.all(attention_mask[:, -1]):
        raise ValueError(
            "each prompt must end in a valid token; left-pad variable-length batches"
        )
    visible = result[:, -model.config.max_seq_len :]
    visible_mask = attention_mask[:, -model.config.max_seq_len :]
    logits, cache = prefill(model, visible, visible_mask)
    finished = torch.zeros(result.shape[0], device=result.device, dtype=torch.bool)

    for step in range(max_new_tokens):
        next_id = sample_next_token(
            logits[:, -1],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            generator=generator,
        )
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
        if step + 1 == max_new_tokens:
            break

        cached_len = cache[0][0].shape[-2]
        if cached_len < model.config.max_seq_len:
            visible_mask = torch.cat(
                (visible_mask, torch.ones_like(next_id, dtype=torch.bool)), dim=1
            )
            logits, cache = decode_one(
                model, next_id, cache, attention_mask=visible_mask
            )
        else:
            # Sliding the window changes which token is position zero. Re-prefill
            # the new window so RoPE positions exactly match ordinary generation.
            visible = result[:, -model.config.max_seq_len :]
            visible_mask = attention_mask[:, -model.config.max_seq_len :]
            logits, cache = prefill(model, visible, visible_mask)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", default="清晨，")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    from train import load_checkpoint

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, _ = load_checkpoint(args.checkpoint, device=device)
    model.eval()
    ids = torch.tensor(
        [tokenizer.encode(args.prompt, add_bos=True)], dtype=torch.long, device=device
    )
    generator = torch.Generator(device=device).manual_seed(args.seed)
    print(f"cached/full max_abs_error={cache_equivalence_error(model, ids):.3e}")
    generated = generate_with_kv_cache(
        model,
        ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_token_id=tokenizer.eos_token_id,
        generator=generator,
    )
    print(tokenizer.decode(generated[0].tolist()))


if __name__ == "__main__":
    main()
