import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from common.my_dl_lib import AdamW, CrossEntropyLoss, Momentum  # noqa: E402
from solutions.block_02_resnet.resnet_numpy_solution import (  # noqa: E402
    SmallResNet,
    accuracy,
    iterate_minibatches,
    normalize,
    one_hot,
    to_nchw,
)


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def load_cifar100(data_dir, train=True, limit=None):
    try:
        from torchvision.datasets import CIFAR100
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "torchvision is required for CIFAR-100. Run: pip install -r requirements.txt"
        ) from exc

    dataset = CIFAR100(root=str(data_dir), train=train, download=True)
    images = to_nchw(dataset.data)
    images = normalize(images, CIFAR100_MEAN, CIFAR100_STD)
    labels = np.asarray(dataset.targets, dtype="int64")
    if limit is not None:
        images = images[:limit]
        labels = labels[:limit]
    return images, labels


def random_horizontal_flip(images, p=0.5, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    mask = rng.random(images.shape[0]) < p
    out = images.copy()
    out[mask] = out[mask, :, :, ::-1]
    return out


def random_crop_with_padding(images, padding=4, crop_size=32, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    n, c, h, w = images.shape
    padded = np.pad(images, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    out = np.empty((n, c, crop_size, crop_size), dtype=images.dtype)
    for i in range(n):
        top = rng.integers(0, h + 2 * padding - crop_size + 1)
        left = rng.integers(0, w + 2 * padding - crop_size + 1)
        out[i] = padded[i, :, top : top + crop_size, left : left + crop_size]
    return out


def augment_batch(images, rng):
    images = random_crop_with_padding(images, padding=4, crop_size=32, rng=rng)
    return random_horizontal_flip(images, p=0.5, rng=rng)


def make_optimizer(name, parameters, lr, weight_decay):
    if name == "momentum":
        return Momentum(parameters, lr=lr, beta=0.9)
    if name == "adamw":
        return AdamW(parameters, lr=lr, weight_decay=weight_decay)
    raise ValueError("optimizer must be momentum or adamw")


def model_state_dict(model):
    return {f"param_{idx}": value.copy() for idx, (value, _) in enumerate(model.parameters())}


def load_model_state_dict(model, state):
    for idx, (value, _) in enumerate(model.parameters()):
        value[...] = state[f"param_{idx}"]


def save_checkpoint(path, model, optimizer, epoch, history):
    path.parent.mkdir(parents=True, exist_ok=True)
    state = model_state_dict(model)
    state["epoch"] = np.asarray(epoch, dtype=np.int64)
    state["history_json"] = np.asarray(json.dumps(history))
    state["optimizer_class"] = np.asarray(optimizer.__class__.__name__)
    if hasattr(optimizer, "velocity"):
        for idx, value in enumerate(optimizer.velocity):
            state[f"optimizer_velocity_{idx}"] = value.copy()
    if hasattr(optimizer, "m"):
        for idx, value in enumerate(optimizer.m):
            state[f"optimizer_m_{idx}"] = value.copy()
    if hasattr(optimizer, "v"):
        for idx, value in enumerate(optimizer.v):
            state[f"optimizer_v_{idx}"] = value.copy()
    if hasattr(optimizer, "t"):
        state["optimizer_t"] = np.asarray(optimizer.t, dtype=np.int64)
    np.savez_compressed(path, **state)


def load_checkpoint(path, model, optimizer):
    state = np.load(path)
    load_model_state_dict(model, state)
    if hasattr(optimizer, "velocity") and "optimizer_velocity_0" in state:
        for idx in range(len(optimizer.velocity)):
            optimizer.velocity[idx][...] = state[f"optimizer_velocity_{idx}"]
    if hasattr(optimizer, "m") and "optimizer_m_0" in state:
        for idx in range(len(optimizer.m)):
            optimizer.m[idx][...] = state[f"optimizer_m_{idx}"]
    if hasattr(optimizer, "v") and "optimizer_v_0" in state:
        for idx in range(len(optimizer.v)):
            optimizer.v[idx][...] = state[f"optimizer_v_{idx}"]
    if hasattr(optimizer, "t") and "optimizer_t" in state:
        optimizer.t = int(state["optimizer_t"])
    history = json.loads(str(state["history_json"])) if "history_json" in state else []
    return int(state["epoch"]), history


def train_epoch(model, loss_fn, optimizer, images, labels, batch_size, epoch, augment):
    model.train()
    rng = np.random.default_rng(epoch)
    losses = []
    correct = 0
    total = 0
    for x, y in iterate_minibatches(images, labels, batch_size, shuffle=True, seed=epoch):
        if augment:
            x = augment_batch(x, rng)
        targets = one_hot(y, num_classes=100)
        logits = model.forward(x)
        loss = loss_fn.forward(logits, targets)
        model.backward(loss_fn.backward())
        optimizer.step()
        losses.append(loss)
        correct += np.sum(np.argmax(logits, axis=1) == y)
        total += y.shape[0]
    return float(np.mean(losses)), float(correct / total)


def evaluate(model, loss_fn, images, labels, batch_size):
    model.eval()
    losses = []
    correct = 0
    total = 0
    for x, y in iterate_minibatches(images, labels, batch_size, shuffle=False):
        targets = one_hot(y, num_classes=100)
        logits = model.forward(x)
        losses.append(loss_fn.forward(logits, targets))
        correct += np.sum(np.argmax(logits, axis=1) == y)
        total += y.shape[0]
    return float(np.mean(losses)), float(correct / total)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the NumPy SmallResNet on CIFAR-100.")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--optimizer", choices=["momentum", "adamw"], default="momentum")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=1000)
    parser.add_argument("--subset-size", type=int, default=None)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "checkpoints" / "cifar100_numpy_resnet.npz")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--channels", type=int, nargs=3, default=(16, 32, 64))
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    if args.subset_size is not None:
        train_images, train_labels = load_cifar100(args.data_dir, train=True, limit=args.subset_size)
        val_limit = min(args.val_limit, args.subset_size)
        val_images, val_labels = load_cifar100(args.data_dir, train=False, limit=val_limit)
        augment = False
    else:
        train_images, train_labels = load_cifar100(args.data_dir, train=True, limit=args.train_limit)
        val_images, val_labels = load_cifar100(args.data_dir, train=False, limit=args.val_limit)
        augment = not args.no_augment

    model = SmallResNet(num_classes=100, channels=tuple(args.channels), blocks_per_stage=(2, 2, 2))
    loss_fn = CrossEntropyLoss()
    optimizer = make_optimizer(args.optimizer, model.parameters(), args.lr, args.weight_decay)
    history = []
    start_epoch = 1

    if args.resume:
        last_epoch, history = load_checkpoint(args.checkpoint, model, optimizer)
        start_epoch = last_epoch + 1
        print(f"resumed checkpoint={args.checkpoint} start_epoch={start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model,
            loss_fn,
            optimizer,
            train_images,
            train_labels,
            args.batch_size,
            epoch,
            augment,
        )
        val_loss, val_acc = evaluate(model, loss_fn, val_images, val_labels, args.batch_size)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        history.append(row)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )
        save_checkpoint(args.checkpoint, model, optimizer, epoch, history)


if __name__ == "__main__":
    main()
