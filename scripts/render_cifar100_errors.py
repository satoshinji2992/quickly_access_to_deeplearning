#!/usr/bin/env python3
"""Render a figure from *real* CIFAR-100 test-set mistakes.

This script deliberately refuses to invent examples.  It runs a strict,
complete ResNet-20 checkpoint on all 10,000 images in the official CIFAR-100
test split, selects mistakes deterministically, and records the run provenance
in the PNG metadata and footer.

Normal usage (the torchvision dataset is verified by its official MD5):

    python scripts/render_cifar100_errors.py \
        --checkpoint /path/to/cifar100_resnet20.pt --download-data

For the repository figure we use the Hugging Face Parquet conversion of the
same official test split.  This avoids downloading the 169 MB archive merely
to regenerate one figure; the exact Parquet SHA-256 is checked:

    python scripts/render_cifar100_errors.py \
        --checkpoint /path/to/cifar100_resnet20-23dac2f1.pt \
        --test-parquet /path/to/cifar100-test.parquet

The reference checkpoint is the public CIFAR-100 ResNet-20 checkpoint from
``chenyaofo/pytorch-cifar-models``.  It reports 68.83% top-1 accuracy and is
available at the immutable release URL in ``REFERENCE_CHECKPOINT_URL``.
``--download-reference-checkpoint`` downloads it to the system temp directory;
it never touches ``checkpoints/`` or another user checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch import nn


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_OUTPUT = REPOSITORY_ROOT / "assets/images/misclassified_examples.png"
TASK_OUTPUT = (
    REPOSITORY_ROOT
    / "exercises/block_02_resnet/task_15_experiment_notes/assets"
    / "misclassified_examples.png"
)

REFERENCE_CHECKPOINT_URL = (
    "https://github.com/chenyaofo/pytorch-cifar-models/releases/download/"
    "resnet/cifar100_resnet20-23dac2f1.pt"
)
REFERENCE_CHECKPOINT_SHA256 = (
    "23dac2f1bc53fd65b9fd0bd30a46f91ab39fa16505430cc7999662237598209d"
)
REFERENCE_PARQUET_SHA256 = (
    "98776c529bb146a9c791229df74a5cf076be9b43d82dbbd334b6a7788d73dc68"
)
DEFAULT_SEED = 20260818
CIFAR100_MEAN = np.asarray((0.5071, 0.4867, 0.4408), dtype=np.float32)
CIFAR100_STD = np.asarray((0.2675, 0.2565, 0.2761), dtype=np.float32)

CIFAR100_CLASSES = (
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee",
    "beetle", "bicycle", "bottle", "bowl", "boy", "bridge", "bus",
    "butterfly", "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab",
    "crocodile", "cup", "dinosaur", "dolphin", "elephant", "flatfish",
    "forest", "fox", "girl", "hamster", "house", "kangaroo", "keyboard",
    "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster", "man",
    "maple_tree", "motorcycle", "mountain", "mouse", "mushroom", "oak_tree",
    "orange", "orchid", "otter", "palm_tree", "pear", "pickup_truck",
    "pine_tree", "plain", "plate", "poppy", "porcupine", "possum", "rabbit",
    "raccoon", "ray", "road", "rocket", "rose", "sea", "seal", "shark",
    "shrew", "skunk", "skyscraper", "snail", "snake", "spider", "squirrel",
    "streetcar", "sunflower", "sweet_pepper", "table", "tank", "telephone",
    "television", "tiger", "tractor", "train", "trout", "tulip", "turtle",
    "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_reference_checkpoint() -> Path:
    """Download the immutable reference weights outside the repository."""
    target = Path(tempfile.gettempdir()) / "cifar100_resnet20-23dac2f1.pt"
    if target.exists() and sha256(target) == REFERENCE_CHECKPOINT_SHA256:
        return target
    temporary = target.with_suffix(".pt.part")
    urllib.request.urlretrieve(REFERENCE_CHECKPOINT_URL, temporary)
    actual = sha256(temporary)
    if actual != REFERENCE_CHECKPOINT_SHA256:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"downloaded checkpoint SHA-256 is {actual}; "
            f"expected {REFERENCE_CHECKPOINT_SHA256}"
        )
    os.replace(temporary, target)
    return target


class BasicBlock(nn.Module):
    """CIFAR ResNet basic block, with names matching the reference weights."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, 1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        identity = inputs
        outputs = self.relu(self.bn1(self.conv1(inputs)))
        outputs = self.bn2(self.conv2(outputs))
        if self.downsample is not None:
            identity = self.downsample(inputs)
        return self.relu(outputs + identity)


class CifarResNet20(nn.Module):
    """The 3-3-3 CIFAR ResNet-20 used by the reference checkpoint."""

    def __init__(self, num_classes: int = 100):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._stage(16, 16, stride=1)
        self.layer2 = self._stage(16, 32, stride=2)
        self.layer3 = self._stage(32, 64, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    @staticmethod
    def _stage(in_channels: int, out_channels: int, stride: int) -> nn.Sequential:
        return nn.Sequential(
            BasicBlock(in_channels, out_channels, stride),
            BasicBlock(out_channels, out_channels),
            BasicBlock(out_channels, out_channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.relu(self.bn1(self.conv1(inputs)))
        outputs = self.layer1(outputs)
        outputs = self.layer2(outputs)
        outputs = self.layer3(outputs)
        outputs = self.avgpool(outputs).flatten(1)
        return self.fc(outputs)


def _extract_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must contain a state-dict mapping")
    for key in ("model_state_dict", "state_dict", "model"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, dict):
            checkpoint = candidate
            break
    if not checkpoint or not all(isinstance(name, str) for name in checkpoint):
        raise TypeError("checkpoint does not contain a recognizable model state dict")
    state = {}
    for name, value in checkpoint.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"checkpoint entry {name!r} is not a tensor")
        state[name.removeprefix("module.")] = value
    return state


def load_model(checkpoint_path: Path, device: torch.device) -> CifarResNet20:
    model = CifarResNet20(num_classes=100)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = _extract_state_dict(checkpoint)
    # Strict loading is intentional: missing BatchNorm running statistics would
    # make the visualization just as misleading as a fabricated sample grid.
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    return model


def load_torchvision_test(data_dir: Path, download: bool) -> tuple[np.ndarray, np.ndarray]:
    from torchvision.datasets import CIFAR100

    dataset = CIFAR100(root=str(data_dir), train=False, download=download)
    if tuple(dataset.classes) != CIFAR100_CLASSES:
        raise ValueError("CIFAR-100 class order does not match the official metadata")
    return np.asarray(dataset.data), np.asarray(dataset.targets, dtype=np.int64)


def load_parquet_test(path: Path) -> tuple[np.ndarray, np.ndarray]:
    actual_sha = sha256(path)
    if actual_sha != REFERENCE_PARQUET_SHA256:
        raise ValueError(
            f"test Parquet SHA-256 is {actual_sha}; expected {REFERENCE_PARQUET_SHA256}"
        )
    try:
        import pyarrow.parquet as parquet
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "--test-parquet needs pyarrow; use the torchvision path instead"
        ) from exc

    table = parquet.read_table(path, columns=["img", "fine_label"])
    images = [
        np.asarray(Image.open(io.BytesIO(item["bytes"])).convert("RGB"), dtype=np.uint8)
        for item in table["img"].to_pylist()
    ]
    return np.stack(images), np.asarray(table["fine_label"], dtype=np.int64)


def validate_test_split(images: np.ndarray, labels: np.ndarray) -> None:
    if images.shape != (10_000, 32, 32, 3) or images.dtype != np.uint8:
        raise ValueError(
            "expected the official CIFAR-100 test split with shape "
            f"(10000, 32, 32, 3) uint8; got {images.shape} {images.dtype}"
        )
    if labels.shape != (10_000,) or set(np.unique(labels)) != set(range(100)):
        raise ValueError("expected 10,000 labels covering all 100 CIFAR-100 classes")
    counts = np.bincount(labels, minlength=100)
    if not np.all(counts == 100):
        raise ValueError("official CIFAR-100 test split must contain 100 images per class")


def iter_batches(total: int, batch_size: int) -> Iterable[slice]:
    for start in range(0, total, batch_size):
        yield slice(start, min(total, start + batch_size))


def infer(
    model: nn.Module,
    images: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    mean = torch.as_tensor(CIFAR100_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.as_tensor(CIFAR100_STD, device=device).view(1, 3, 1, 1)
    predictions = []
    confidences = []
    with torch.inference_mode():
        for indices in iter_batches(len(images), batch_size):
            batch = torch.from_numpy(images[indices].copy()).to(device=device)
            batch = batch.permute(0, 3, 1, 2).float().div_(255.0)
            probabilities = ((batch - mean) / std)
            probabilities = model(probabilities).softmax(dim=1)
            confidence, prediction = probabilities.max(dim=1)
            predictions.append(prediction.cpu().numpy())
            confidences.append(confidence.cpu().numpy())
    return np.concatenate(predictions), np.concatenate(confidences)


def select_errors(
    labels: np.ndarray,
    predictions: np.ndarray,
    confidences: np.ndarray,
    count: int,
    seed: int,
) -> np.ndarray:
    """Choose varied, reproducible mistakes without hand-picking examples."""
    errors = np.flatnonzero(labels != predictions)
    if len(errors) < count:
        raise ValueError(f"model produced only {len(errors)} errors; need {count}")

    # Sample from the upper half by confidence so each card demonstrates a
    # meaningful wrong decision, while seeded shuffling avoids cherry-picking
    # particular objects.  Prefer distinct true and predicted classes.
    threshold = np.quantile(confidences[errors], 0.50)
    candidates = errors[confidences[errors] >= threshold]
    rng = np.random.default_rng(seed)
    candidates = candidates[rng.permutation(len(candidates))]
    selected: list[int] = []
    true_classes: set[int] = set()
    predicted_classes: set[int] = set()
    for index in candidates:
        true_class = int(labels[index])
        predicted_class = int(predictions[index])
        if true_class in true_classes or predicted_class in predicted_classes:
            continue
        selected.append(int(index))
        true_classes.add(true_class)
        predicted_classes.add(predicted_class)
        if len(selected) == count:
            break
    if len(selected) < count:
        for index in candidates:
            if int(index) not in selected:
                selected.append(int(index))
            if len(selected) == count:
                break
    return np.asarray(selected, dtype=np.int64)


def render(
    images: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    confidences: np.ndarray,
    selected: np.ndarray,
    accuracy: float,
    checkpoint_sha: str,
    seed: int,
    output_paths: tuple[Path, ...],
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "semibold",
            "figure.facecolor": "#F7F5F0",
        }
    )
    figure = plt.figure(figsize=(12, 7.5), dpi=160, facecolor="#F7F5F0")
    grid = figure.add_gridspec(
        2, 4, left=0.045, right=0.955, top=0.82, bottom=0.18, wspace=0.16, hspace=0.38
    )

    figure.text(
        0.05,
        0.925,
        "CIFAR-100  /  real test errors",
        fontsize=25,
        color="#17233B",
        weight="bold",
        ha="left",
    )
    figure.text(
        0.05,
        0.875,
        f"ResNet-20   ·   official test split   ·   top-1 {accuracy:.2%}",
        fontsize=12.5,
        color="#5E6879",
        ha="left",
    )

    for position, index in enumerate(selected):
        axis = figure.add_subplot(grid[position // 4, position % 4])
        axis.set_facecolor("#FFFFFF")
        axis.imshow(images[index], interpolation="nearest")
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color("#DDD8CE")
            spine.set_linewidth(1.2)
        true_name = CIFAR100_CLASSES[int(labels[index])].replace("_", " ")
        predicted_name = CIFAR100_CLASSES[int(predictions[index])].replace("_", " ")
        axis.text(
            0.0,
            -0.10,
            f"true  {true_name}",
            transform=axis.transAxes,
            fontsize=10.5,
            color="#246B5A",
            weight="semibold",
            ha="left",
            va="top",
        )
        axis.text(
            0.0,
            -0.20,
            f"pred  {predicted_name}  ·  {confidences[index]:.0%}",
            transform=axis.transAxes,
            fontsize=10.5,
            color="#B84B3E",
            ha="left",
            va="top",
        )

    footer = (
        f"10,000 test images  ·  selection seed {seed}  ·  "
        f"checkpoint SHA-256 {checkpoint_sha[:12]}…  ·  no synthetic samples"
    )
    figure.text(0.05, 0.045, footer, fontsize=9.5, color="#6D7480", ha="left")
    metadata = {
        "Title": "CIFAR-100 real test-set misclassifications",
        "Description": json.dumps(
            {
                "dataset": "official CIFAR-100 test split",
                "dataset_size": 10_000,
                "model": "CIFAR ResNet-20",
                "checkpoint_sha256": checkpoint_sha,
                "accuracy": accuracy,
                "selection_seed": seed,
                "test_indices": selected.tolist(),
                "true_labels": labels[selected].tolist(),
                "predicted_labels": predictions[selected].tolist(),
                "synthetic_samples": False,
            },
            sort_keys=True,
        ),
    }
    for output_path in output_paths:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            output_path,
            dpi=160,
            facecolor=figure.get_facecolor(),
            metadata=metadata,
        )
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--download-reference-checkpoint",
        action="store_true",
        help="download exact public weights to the system temp directory",
    )
    parser.add_argument("--data-dir", type=Path, default=REPOSITORY_ROOT / "data")
    parser.add_argument("--download-data", action="store_true")
    parser.add_argument(
        "--test-parquet",
        type=Path,
        help="verified Hugging Face Parquet conversion of the official test split",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--count", type=int, default=8, choices=(4, 8))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "mps", "cuda"),
        help="CPU is the reproducible default used for the committed figure",
    )
    parser.add_argument(
        "--output",
        type=Path,
        action="append",
        help="override output path; repeat to write more than one copy",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.download_reference_checkpoint:
        if args.checkpoint is not None:
            raise ValueError("choose --checkpoint or --download-reference-checkpoint, not both")
        checkpoint_path = download_reference_checkpoint()
    elif args.checkpoint is not None:
        checkpoint_path = args.checkpoint
    else:
        raise ValueError("provide --checkpoint or --download-reference-checkpoint")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    if args.test_parquet is not None:
        images, labels = load_parquet_test(args.test_parquet)
    else:
        images, labels = load_torchvision_test(args.data_dir, args.download_data)
    validate_test_split(images, labels)

    device = torch.device(args.device)
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model = load_model(checkpoint_path, device)
    predictions, confidences = infer(model, images, device, args.batch_size)
    accuracy = float(np.mean(predictions == labels))
    selected = select_errors(labels, predictions, confidences, args.count, args.seed)
    checkpoint_sha = sha256(checkpoint_path)
    outputs = tuple(args.output) if args.output else (CANONICAL_OUTPUT, TASK_OUTPUT)
    render(
        images,
        labels,
        predictions,
        confidences,
        selected,
        accuracy,
        checkpoint_sha,
        args.seed,
        outputs,
    )
    print(f"dataset=official CIFAR-100 test images={len(images)}")
    print(f"model=ResNet-20 checkpoint_sha256={checkpoint_sha}")
    print(f"top1={accuracy:.4f} seed={args.seed} indices={selected.tolist()}")
    for output in outputs:
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
