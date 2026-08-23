"""Render real artifacts for the homepage results strip.

- Circle classifier decision boundary (same 2-4-4-2 net as task_01, trained here)
- MiniMind generation samples (trains 80 steps, then greedy + sampled generation)
- KV cache equivalence error (task_30 on the same checkpoint)

Outputs into site/static/assets/results/.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "site" / "static" / "assets" / "results"
ROOT_BIN = sys.executable


def circle_boundary() -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(7)
    n = 800
    x = rng.uniform(-1.6, 1.6, n)
    y = rng.uniform(-1.6, 1.6, n)
    label = (x * x + y * y <= 1.0).astype(int)
    X = np.stack([x, y], axis=1)
    Y = np.eye(2)[label]

    def init(fan_in, fan_out):
        return rng.normal(0, np.sqrt(2 / fan_in), (fan_in, fan_out))

    W1, b1 = init(2, 4), np.zeros((1, 4))
    W2, b2 = init(4, 4), np.zeros((1, 4))
    W3, b3 = init(4, 2), np.zeros((1, 2))
    lr = 0.12
    for _ in range(1200):
        h1 = np.maximum(0, X @ W1 + b1)
        h2 = np.maximum(0, h1 @ W2 + b2)
        p = np.exp(h2 @ W3 + b3 - np.amax(h2 @ W3 + b3, axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        dz3 = (p - Y) / len(X)
        dW3, db3 = h2.T @ dz3, dz3.sum(0, keepdims=True)
        dh2 = dz3 @ W3.T
        dz2 = dh2 * (h2 > 0)
        dW2, db2 = h1.T @ dz2, dz2.sum(0, keepdims=True)
        dh1 = dz2 @ W2.T
        dz1 = dh1 * (h1 > 0)
        dW1, db1 = X.T @ dz1, dz1.sum(0, keepdims=True)
        for w, dw, b, db in ((W1, dW1, b1, db1), (W2, dW2, b2, db2), (W3, dW3, b3, db3)):
            w -= lr * dw
            b -= lr * db
    acc = ((p.argmax(1)) == label).mean()

    gs = 240
    xa = np.linspace(-1.6, 1.6, gs)
    ga = np.stack(np.meshgrid(xa, xa), axis=-1).reshape(-1, 2)
    h1 = np.maximum(0, ga @ W1 + b1)
    h2 = np.maximum(0, h1 @ W2 + b2)
    lg = h2 @ W3 + b3
    grid = lg[:, 1] - lg[:, 0]
    grid = grid.reshape(gs, gs)

    fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=150)
    ax.imshow(grid, extent=(-1.6, 1.6, -1.6, 1.6), origin="lower",
              cmap="RdYlBu_r", alpha=0.35, vmin=-6, vmax=6)
    ax.scatter(x[label == 1], y[label == 1], s=5, c="#0b63f3", linewidths=0)
    ax.scatter(x[label == 0], y[label == 0], s=5, c="#c8ff47", linewidths=0)
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), color="#071321", lw=1.4, ls="--", label="真实边界")
    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#071321"); s.set_linewidth(1.6)
    ax.legend(loc="upper right", fontsize=7, frameon=False)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "circle_boundary.png", bbox_inches="tight", facecolor="#fffef9")
    plt.close(fig)
    return {"circle_val_acc": round(float(acc), 3)}


def minimind() -> dict:
    ckpt = Path("/tmp/minimind_home.pt")
    train = ROOT / "exercises/block_03_transformer/task_28_next_token_training/train.py"
    gen = ROOT / "exercises/block_03_transformer/task_29_generate_sampling/generate.py"
    kv = ROOT / "exercises/block_03_transformer/task_30_kv_cache/kv_cache.py"

    def run(*args):
        return subprocess.run([ROOT_BIN, *(str(a) for a in args)],
                              capture_output=True, text=True, check=True).stdout

    run(train, "--steps", "120", "--checkpoint", str(ckpt))
    greedy = run(gen, "--checkpoint", str(ckpt), "--prompt", "清晨，", "--max-new-tokens", "24")
    sampled = run(gen, "--checkpoint", str(ckpt), "--prompt", "清晨，", "--max-new-tokens", "24",
                  "--temperature", "0.8", "--top-k", "20", "--top-p", "0.9", "--seed", "3")
    kvout = run(kv, "--checkpoint", str(ckpt), "--prompt", "清晨，", "--max-new-tokens", "20")
    kv_line = next((l for l in kvout.splitlines() if "max_abs_error" in l), "")
    return {
        "greedy": greedy.strip().splitlines()[-1] if greedy.strip() else "",
        "sampled": sampled.strip().splitlines()[-1] if sampled.strip() else "",
        "kv_error": kv_line.replace("cached/full ", ""),
    }


if __name__ == "__main__":
    data = {}
    data.update(circle_boundary())
    try:
        data.update(minimind())
    except Exception as exc:  # torch 环境缺失时不阻塞边界图
        data["minimind_error"] = str(exc)[:120]
    (OUT / "results.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(data, ensure_ascii=False, indent=1))
