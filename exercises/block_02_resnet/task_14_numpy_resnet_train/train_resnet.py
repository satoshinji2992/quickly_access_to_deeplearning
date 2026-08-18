"""A narrow but complete NumPy ResNet training program.

Use ``--synthetic`` for a fast, download-free smoke test.  The normal path
uses an isolated CIFAR-100 train/validation/test split from task 10.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from common.my_dl_lib import CrossEntropyLoss, Linear, Momentum, ReLU  # noqa: E402
from exercises.block_02_resnet.task_10_image_data_pipeline.data_pipeline import (  # noqa: E402
    iterate_minibatches,
    load_cifar100_splits,
    random_crop_with_padding,
    random_horizontal_flip,
    stratified_train_val_split,
)
from exercises.block_02_resnet.task_11_conv2d_im2col.conv2d import Conv2D  # noqa: E402
from exercises.block_02_resnet.task_12_pooling_and_bn.layers import (  # noqa: E402
    BatchNorm2D,
    GlobalAvgPool2D,
)
from exercises.block_02_resnet.task_13_residual_block.residual_block import (  # noqa: E402
    BasicBlock,
)


def _parameters(layer):
    return layer.parameters() if hasattr(layer, "parameters") else []


def _set_mode(layer, training):
    method = getattr(layer, "train" if training else "eval", None)
    if method is not None:
        method()


class SmallResNet:
    """A CIFAR-sized ResNet assembled from tasks 11--13."""

    def __init__(
        self,
        num_classes=100,
        channels=(16, 32, 64),
        blocks_per_stage=(2, 2, 2),
    ):
        channels = tuple(int(value) for value in channels)
        blocks_per_stage = tuple(int(value) for value in blocks_per_stage)
        if len(channels) != len(blocks_per_stage) or not channels:
            raise ValueError("channels and blocks_per_stage must have equal non-zero length")
        if any(value <= 0 for value in channels + blocks_per_stage):
            raise ValueError("channel and block counts must be positive")
        self.num_classes = int(num_classes)
        if self.num_classes <= 1:
            raise ValueError("num_classes must be greater than one")

        self.stem_conv = Conv2D(3, channels[0], 3, stride=1, padding=1)
        self.stem_bn = BatchNorm2D(channels[0])
        self.stem_relu = ReLU()
        # Keep the original reference-solution attribute available for readers
        # who inspect ``model.stem[1].running_mean`` in checkpoint experiments.
        self.stem = [self.stem_conv, self.stem_bn, self.stem_relu]
        self.blocks = []
        in_channels = channels[0]
        for stage_index, (out_channels, block_count) in enumerate(
            zip(channels, blocks_per_stage)
        ):
            for block_index in range(block_count):
                stride = 2 if stage_index > 0 and block_index == 0 else 1
                self.blocks.append(BasicBlock(in_channels, out_channels, stride=stride))
                in_channels = out_channels
        self.pool = GlobalAvgPool2D()
        self.fc = Linear(channels[-1], self.num_classes)
        self.training = True

    def forward(self, x):
        x = self.stem_conv.forward(x)
        x = self.stem_bn.forward(x)
        x = self.stem_relu.forward(x)
        for block in self.blocks:
            x = block.forward(x)
        x = self.pool.forward(x)
        return self.fc.forward(x)

    def backward(self, dout):
        dout = self.fc.backward(dout)
        dout = self.pool.backward(dout)
        for block in reversed(self.blocks):
            dout = block.backward(dout)
        dout = self.stem_relu.backward(dout)
        dout = self.stem_bn.backward(dout)
        return self.stem_conv.backward(dout)

    def parameters(self):
        return [(value, gradient) for _, value, gradient in self.named_parameters()]

    def named_parameters(self, prefix=""):
        named = []
        stem_layers = (("stem_conv", self.stem_conv), ("stem_bn", self.stem_bn))
        for name, layer in stem_layers:
            child_prefix = ".".join(part for part in (prefix, name) if part)
            named.extend(layer.named_parameters(child_prefix))
        for index, block in enumerate(self.blocks):
            child_prefix = ".".join(
                part for part in (prefix, f"blocks.{index}") if part
            )
            named.extend(block.named_parameters(child_prefix))
        fc_prefix = f"{prefix}." if prefix else ""
        named.extend(
            [
                (f"{fc_prefix}fc.W", self.fc.W, self.fc.dW),
                (f"{fc_prefix}fc.b", self.fc.b, self.fc.db),
            ]
        )
        return named

    def named_buffers(self, prefix=""):
        named = []
        stem_prefix = ".".join(part for part in (prefix, "stem_bn") if part)
        named.extend(self.stem_bn.named_buffers(stem_prefix))
        for index, block in enumerate(self.blocks):
            child_prefix = ".".join(
                part for part in (prefix, f"blocks.{index}") if part
            )
            named.extend(block.named_buffers(child_prefix))
        return named

    def state_dict(self):
        state = {
            name: value.copy() for name, value, _ in self.named_parameters()
        }
        state.update({name: value.copy() for name, value in self.named_buffers()})
        return state

    def load_state_dict(self, state, strict=True):
        destinations = {
            name: value for name, value, _ in self.named_parameters()
        }
        destinations.update({name: value for name, value in self.named_buffers()})
        missing = sorted(set(destinations) - set(state))
        unexpected = sorted(set(state) - set(destinations))
        if strict and (missing or unexpected):
            raise KeyError(f"state mismatch: missing={missing}, unexpected={unexpected}")
        for name, destination in destinations.items():
            if name not in state:
                continue
            source = np.asarray(state[name])
            if source.shape != destination.shape:
                raise ValueError(
                    f"state[{name!r}] has shape {source.shape}; expected {destination.shape}"
                )
            destination[...] = source
        return {"missing_keys": missing, "unexpected_keys": unexpected}

    def train(self):
        self.training = True
        for layer in (self.stem_conv, self.stem_bn, self.stem_relu, self.pool, self.fc):
            _set_mode(layer, training=True)
        for block in self.blocks:
            block.train()
        return self

    def eval(self):
        self.training = False
        for layer in (self.stem_conv, self.stem_bn, self.stem_relu, self.pool, self.fc):
            _set_mode(layer, training=False)
        for block in self.blocks:
            block.eval()
        return self


def one_hot(labels, num_classes):
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("labels must be a one-dimensional integer array")
    if np.any(labels < 0) or np.any(labels >= num_classes):
        raise ValueError(f"labels must be in [0, {num_classes})")
    targets = np.zeros((labels.shape[0], num_classes), dtype=np.float64)
    targets[np.arange(labels.shape[0]), labels] = 1.0
    return targets


def accuracy(logits, labels):
    return float(np.mean(np.argmax(logits, axis=1) == labels))


def augment_batch(images, rng):
    images = random_crop_with_padding(images, padding=4, crop_size=images.shape[-1], rng=rng)
    return random_horizontal_flip(images, p=0.5, rng=rng)


def train_epoch(
    model,
    loss_fn,
    optimizer,
    images,
    labels,
    batch_size,
    seed=0,
    augment=False,
):
    model.train()
    rng = np.random.default_rng(seed)
    loss_total = 0.0
    correct = 0
    sample_count = 0
    for batch_images, batch_labels in iterate_minibatches(
        images, labels, batch_size, shuffle=True, seed=seed
    ):
        if augment:
            batch_images = augment_batch(batch_images, rng)
        logits = model.forward(batch_images)
        loss = loss_fn.forward(logits, one_hot(batch_labels, model.num_classes))
        model.backward(loss_fn.backward())
        optimizer.step()
        # CrossEntropyLoss returns a per-batch mean.  Weight it by the actual
        # batch length so a short final batch does not count as much as a full
        # one in the epoch metric.
        loss_total += float(loss) * batch_labels.shape[0]
        correct += int(np.sum(np.argmax(logits, axis=1) == batch_labels))
        sample_count += batch_labels.shape[0]
    return float(loss_total / sample_count), float(correct / sample_count)


def evaluate(model, loss_fn, images, labels, batch_size):
    model.eval()
    loss_total = 0.0
    correct = 0
    sample_count = 0
    for batch_images, batch_labels in iterate_minibatches(
        images, labels, batch_size, shuffle=False
    ):
        logits = model.forward(batch_images)
        batch_loss = loss_fn.forward(
            logits, one_hot(batch_labels, model.num_classes)
        )
        loss_total += float(batch_loss) * batch_labels.shape[0]
        correct += int(np.sum(np.argmax(logits, axis=1) == batch_labels))
        sample_count += batch_labels.shape[0]
    return float(loss_total / sample_count), float(correct / sample_count)


def make_synthetic_splits(sample_count=96, num_classes=4, image_size=8, seed=0):
    """Create a learnable image dataset for the no-download smoke path."""
    if sample_count < num_classes * 4:
        raise ValueError("sample_count is too small for stratified train/val/test splits")
    rng = np.random.default_rng(seed)
    labels = np.arange(sample_count, dtype=np.int64) % num_classes
    rng.shuffle(labels)
    images = rng.normal(0.0, 0.15, (sample_count, 3, image_size, image_size))
    for index, label in enumerate(labels):
        channel = label % 3
        stripe = label % image_size
        images[index, channel, stripe : stripe + 1, :] += 1.5
    (train_images, train_labels), (heldout_images, heldout_labels) = (
        stratified_train_val_split(images, labels, val_fraction=0.25, seed=seed)
    )
    (val_images, val_labels), (test_images, test_labels) = stratified_train_val_split(
        heldout_images, heldout_labels, val_fraction=0.5, seed=seed + 1
    )
    return (
        (train_images, train_labels),
        (val_images, val_labels),
        (test_images, test_labels),
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train a small NumPy ResNet.")
    parser.add_argument("--synthetic", action="store_true", help="run without downloading data")
    parser.add_argument("--data-dir", type=Path, default=REPOSITORY_ROOT / "data")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-limit", type=int, default=500)
    parser.add_argument("--val-limit", type=int, default=500)
    parser.add_argument("--test-limit", type=int, default=500)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--channels", type=int, nargs=3, default=(8, 16, 32))
    parser.add_argument("--blocks", type=int, nargs=3, default=(1, 1, 1))
    parser.add_argument("--no-augment", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive")
    np.random.seed(args.seed)
    if args.synthetic:
        splits = make_synthetic_splits(seed=args.seed)
        num_classes = 4
        augment = False
    else:
        splits = load_cifar100_splits(
            args.data_dir,
            val_size=args.val_size,
            seed=args.seed,
            train_limit=args.train_limit,
            val_limit=args.val_limit,
            test_limit=args.test_limit,
        )
        num_classes = 100
        augment = not args.no_augment

    train, validation, test = splits
    model = SmallResNet(
        num_classes=num_classes,
        channels=tuple(args.channels),
        blocks_per_stage=tuple(args.blocks),
    )
    loss_fn = CrossEntropyLoss()
    optimizer = Momentum(model.parameters(), lr=args.lr, beta=0.9)
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_epoch(
            model,
            loss_fn,
            optimizer,
            *train,
            batch_size=args.batch_size,
            seed=args.seed + epoch,
            augment=augment,
        )
        val_loss, val_accuracy = evaluate(
            model, loss_fn, *validation, batch_size=args.batch_size
        )
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} train_acc={train_accuracy:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.3f}"
        )
    test_loss, test_accuracy = evaluate(model, loss_fn, *test, batch_size=args.batch_size)
    print(f"test_loss={test_loss:.4f} test_acc={test_accuracy:.3f}")
    return model


if __name__ == "__main__":
    main()
