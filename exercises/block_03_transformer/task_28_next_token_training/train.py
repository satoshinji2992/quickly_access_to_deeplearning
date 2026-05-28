import torch

from pathlib import Path
import sys

CORE = Path(__file__).resolve().parents[1] / "task_27_minimind_core"
sys.path.append(str(CORE))

from minimind_core import MiniMindConfig, MiniMindCore  # noqa: E402


def make_toy_batch(batch_size=8, seq_len=32, vocab_size=128, device="cpu"):
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    return x[:, :-1], x[:, 1:]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = MiniMindConfig(vocab_size=128, dim=128, n_layers=2, n_heads=4, n_kv_heads=2)
    model = MiniMindCore(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for step in range(20):
        input_ids, labels = make_toy_batch(device=device)
        _, loss = model(input_ids, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"step={step} loss={loss.item():.4f}")


if __name__ == "__main__":
    main()
