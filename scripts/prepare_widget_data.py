"""Generate static data assets for the interactive widgets.

Outputs into site/static/assets/widgets/:
  cifar8.png            8 real CIFAR-100 thumbnails as a 8x1 sprite (256x32)
  token_embeddings.json synthetic-but-clustered token vectors, PCA 8d -> 3d

The outputs are committed; rerun only when changing widget data needs.
cifar8.png requires the local CIFAR-100 tarball (gitignored); token
embeddings are pure numpy and always regenerate.
"""
from __future__ import annotations

import json
import tarfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "site" / "static" / "assets" / "widgets"
TAR = ROOT / "data" / "cifar-100-python.tar.gz"


def make_cifar_sprite() -> None:
    try:
        from PIL import Image
    except ImportError:
        print("PIL unavailable; skipping cifar8.png")
        return
    if not TAR.exists():
        print("CIFAR tarball missing; skipping cifar8.png")
        return
    with tarfile.open(TAR) as tar:
        member = tar.getmember("cifar-100-python/train")
        data = pickle_load(tar.extractfile(member))
    images = data[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.uint8)
    labels = np.array(data[b"fine_labels"])
    # Pick one image from 8 different classes for a varied shelf.
    chosen = []
    for label in sorted(set(labels.tolist())):
        if len(chosen) >= 8:
            break
        chosen.append(int(np.where(labels == label)[0][0]))
    sprite = np.concatenate([images[i] for i in chosen], axis=1)
    OUT.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sprite).save(OUT / "cifar8.png")
    print(f"cifar8.png: {sprite.shape} from classes {sorted(set(labels[chosen].tolist()))}")


def pickle_load(fileobj):
    import pickle

    return pickle.load(fileobj, encoding="bytes")


TOKEN_CLUSTERS = {
    "动物": ["猫", "狗", "鱼", "鸟", "虫"],
    "动作": ["跑", "吃", "睡", "看", "追"],
    "形容": ["小", "大", "红", "快", "冷"],
    "虚词": ["的", "了", "在", "是", "和"],
}


def make_token_embeddings() -> None:
    rng = np.random.default_rng(7)
    centers = rng.normal(scale=2.2, size=(len(TOKEN_CLUSTERS), 8))
    tokens, vecs, clusters = [], [], []
    for ci, (name, words) in enumerate(TOKEN_CLUSTERS.items()):
        for w in words:
            tokens.append(w)
            vecs.append(centers[ci] + rng.normal(scale=0.5, size=8))
            clusters.append(name)
    X = np.array(vecs)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    # PCA to 3d.
    centered = X - X.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    xyz = centered @ vt[:3].T
    payload = {
        "note": "合成向量：同一语义簇共享中心 + 噪声，单位化后 PCA 降到 3 维。",
        "tokens": tokens,
        "vec8": [[round(v, 4) for v in row] for row in X.tolist()],
        "xyz": [[round(v, 4) for v in row] for row in xyz.tolist()],
        "clusters": clusters,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "token_embeddings.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    print(f"token_embeddings.json: {len(tokens)} tokens")


if __name__ == "__main__":
    try:
        make_cifar_sprite()
    except Exception as exc:  # 残缺的 tar 包不应中断 token 数据生成
        print(f"cifar8.png skipped ({exc}); sprite 已存在则继续沿用")
    make_token_embeddings()
