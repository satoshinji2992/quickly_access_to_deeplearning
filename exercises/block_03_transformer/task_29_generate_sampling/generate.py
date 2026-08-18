"""Greedy, temperature, top-k and nucleus generation for task 28 checkpoints."""

import argparse
from pathlib import Path
import sys

import torch


BLOCK = Path(__file__).resolve().parents[1]
TRAINING = BLOCK / "task_28_next_token_training"
if str(TRAINING) not in sys.path:
    sys.path.insert(0, str(TRAINING))

def sample_next_token(
    logits, temperature=1.0, top_k=None, top_p=None, generator=None
):
    """Sample one id from ``(batch, vocab)`` logits; temperature 0 is greedy."""
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
        # Shift right so the first token crossing p is retained. This also
        # guarantees that every row keeps at least one finite candidate.
        remove_sorted[:, 1:] = remove_sorted[:, :-1].clone()
        remove_sorted[:, 0] = False
        remove = torch.zeros_like(remove_sorted).scatter(
            dim=-1, index=sorted_indices, src=remove_sorted
        )
        filtered = filtered.masked_fill(remove, float("-inf"))
    probabilities = torch.softmax(filtered, dim=-1)
    return torch.multinomial(probabilities, 1, generator=generator)


@torch.no_grad()
def generate(
    model,
    input_ids,
    max_new_tokens,
    temperature=1.0,
    top_k=None,
    top_p=None,
    eos_token_id=None,
    generator=None,
    attention_mask=None,
):
    """Readable non-cached reference loop; task 30 removes repeated work."""
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
        pad_id = model.config.pad_token_id
        attention_mask = (
            torch.ones_like(result, dtype=torch.bool)
            if pad_id is None
            else result.ne(pad_id)
        )
    elif attention_mask.shape != result.shape:
        raise ValueError("attention_mask must have the same shape as input_ids")
    else:
        attention_mask = attention_mask.to(device=result.device, dtype=torch.bool)
    if not torch.all(attention_mask[:, -1]):
        raise ValueError(
            "each prompt must end in a valid token; left-pad variable-length batches"
        )
    finished = torch.zeros(result.shape[0], device=result.device, dtype=torch.bool)
    for _ in range(max_new_tokens):
        context = result[:, -model.config.max_seq_len :]
        context_mask = attention_mask[:, -model.config.max_seq_len :]
        logits, _ = model(context, attention_mask=context_mask)
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
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", default="清晨，")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--top-p", type=float, default=None, help="optional nucleus threshold in (0, 1]"
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    # CLI-only dependency: keeping it local makes this module safe to import in
    # a larger pytest process that may already contain an unrelated `train`.
    from train import load_checkpoint

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, checkpoint = load_checkpoint(args.checkpoint, device=device)
    model.eval()
    prompt_ids = tokenizer.encode(args.prompt, add_bos=True)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    output = generate(
        model,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_token_id=tokenizer.eos_token_id,
        generator=generator,
    )
    print(f"checkpoint_step={checkpoint['step']}")
    print(tokenizer.decode(output[0].tolist()))


if __name__ == "__main__":
    main()
