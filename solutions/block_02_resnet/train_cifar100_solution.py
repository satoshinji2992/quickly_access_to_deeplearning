"""Train and checkpoint the NumPy ResNet on isolated CIFAR-100 splits."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from common.my_dl_lib import AdamW, CrossEntropyLoss, Momentum  # noqa: E402
from exercises.block_02_resnet.task_10_image_data_pipeline.data_pipeline import (  # noqa: E402
    CIFAR100_MEAN,
    CIFAR100_STD,
    load_cifar100_splits,
    normalize,
    stratified_subset,
    to_nchw,
)
from exercises.block_02_resnet.task_14_numpy_resnet_train.train_resnet import (  # noqa: E402
    SmallResNet,
    evaluate,
    train_epoch,
)


CHECKPOINT_VERSION = 2

# These options determine the model, optimizer trajectory, data split, or
# augmentation stream.  A faithful resume restores them from the checkpoint
# before any of those objects are constructed.  ``epochs`` remains a command-
# line target so a run can be extended, and ``data_dir`` may move with the repo.
RESUME_CONFIG_FIELDS = (
    "batch_size",
    "lr",
    "optimizer",
    "weight_decay",
    "val_size",
    "val_limit",
    "test_limit",
    "no_augment",
    "seed",
    "channels",
    "blocks",
)


def load_cifar100(data_dir, train=True, limit=None, download=True):
    """Backward-compatible loader for one official CIFAR-100 split.

    Training code should use :func:`load_cifar100_splits`, which creates a
    validation set without consuming the official test set.
    """
    try:
        from torchvision.datasets import CIFAR100
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "torchvision is required for CIFAR-100; install requirements.txt"
        ) from exc
    dataset = CIFAR100(root=str(data_dir), train=train, download=download)
    images = np.asarray(dataset.data)
    labels = np.asarray(dataset.targets, dtype=np.int64)
    if limit is not None:
        images, labels = stratified_subset(images, labels, limit, seed=0)
    return normalize(to_nchw(images), CIFAR100_MEAN, CIFAR100_STD), labels


def make_optimizer(name, parameters, lr, weight_decay=0.0):
    if name == "momentum":
        return Momentum(parameters, lr=lr, beta=0.9)
    if name == "adamw":
        return AdamW(parameters, lr=lr, weight_decay=weight_decay)
    raise ValueError("optimizer must be 'momentum' or 'adamw'")


def model_state_dict(model):
    """Copy trainable parameters and non-trainable buffers by stable name."""
    if hasattr(model, "state_dict"):
        return model.state_dict()
    if not hasattr(model, "named_parameters"):
        return {
            f"param_{index}": value.copy()
            for index, (value, _) in enumerate(model.parameters())
        }
    state = {
        name: value.copy() for name, value, _ in model.named_parameters()
    }
    if hasattr(model, "named_buffers"):
        state.update({name: value.copy() for name, value in model.named_buffers()})
    return state


def load_model_state_dict(model, state, strict=True):
    """Load model state in place so optimizer-held references stay valid."""
    if hasattr(model, "load_state_dict"):
        return model.load_state_dict(state, strict=strict)
    if hasattr(model, "named_parameters"):
        destinations = {
            name: value for name, value, _ in model.named_parameters()
        }
        if hasattr(model, "named_buffers"):
            destinations.update({name: value for name, value in model.named_buffers()})
    else:
        destinations = {
            f"param_{index}": value
            for index, (value, _) in enumerate(model.parameters())
        }
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


def _json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _optimizer_hyperparameters(optimizer):
    names = (
        "lr",
        "beta",
        "beta1",
        "beta2",
        "rho",
        "eps",
        "weight_decay",
    )
    return {
        name: _json_ready(getattr(optimizer, name))
        for name in names
        if hasattr(optimizer, name)
    }


def _optimizer_array_state(optimizer):
    arrays = {}
    for attribute in ("velocity", "m", "v", "cache"):
        if hasattr(optimizer, attribute):
            for index, value in enumerate(getattr(optimizer, attribute)):
                arrays[f"{attribute}.{index}"] = np.asarray(value).copy()
    return arrays


def _restore_optimizer(
    optimizer,
    optimizer_class,
    hyperparameters,
    arrays,
    step,
    strict=True,
):
    actual_class = optimizer.__class__.__name__
    if strict and optimizer_class != actual_class:
        raise ValueError(
            f"checkpoint optimizer is {optimizer_class}, but destination is {actual_class}"
        )
    for name, value in hyperparameters.items():
        if hasattr(optimizer, name):
            setattr(optimizer, name, value)
        elif strict:
            raise KeyError(f"destination optimizer has no hyperparameter {name!r}")
    for attribute in ("velocity", "m", "v", "cache"):
        if not hasattr(optimizer, attribute):
            continue
        destinations = getattr(optimizer, attribute)
        expected_names = [f"{attribute}.{index}" for index in range(len(destinations))]
        missing = [name for name in expected_names if name not in arrays]
        if strict and missing:
            raise KeyError(f"optimizer state is missing {missing}")
        for index, destination in enumerate(destinations):
            name = f"{attribute}.{index}"
            if name not in arrays:
                continue
            source = np.asarray(arrays[name])
            if source.shape != destination.shape:
                raise ValueError(
                    f"optimizer state {name!r} has shape {source.shape}; "
                    f"expected {destination.shape}"
                )
            destination[...] = source
    if hasattr(optimizer, "t") and step is not None:
        optimizer.t = int(step)


def save_checkpoint(path, model, optimizer, epoch, history, config=None):
    """Atomically save model, BN buffers, optimizer, config, and history."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "checkpoint_version": np.asarray(CHECKPOINT_VERSION, dtype=np.int64),
        "epoch": np.asarray(int(epoch), dtype=np.int64),
        "history_json": np.asarray(json.dumps(_json_ready(history), ensure_ascii=False)),
        "config_json": np.asarray(json.dumps(_json_ready(config or {}), ensure_ascii=False)),
        "optimizer_class": np.asarray(optimizer.__class__.__name__),
        "optimizer_config_json": np.asarray(
            json.dumps(_optimizer_hyperparameters(optimizer), ensure_ascii=False)
        ),
        "model_training": np.asarray(bool(getattr(model, "training", True))),
    }
    if hasattr(optimizer, "t"):
        state["optimizer_step"] = np.asarray(int(optimizer.t), dtype=np.int64)
    for name, value in model_state_dict(model).items():
        state[f"model::{name}"] = value
    for name, value in _optimizer_array_state(optimizer).items():
        state[f"optimizer::{name}"] = value

    temporary_path = path.with_name(f".{path.name}.tmp.npz")
    np.savez_compressed(temporary_path, **state)
    temporary_path.replace(path)


def read_checkpoint_config(path):
    """Read checkpoint metadata without constructing a destination model."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        version = int(archive["checkpoint_version"].item())
        if version != CHECKPOINT_VERSION:
            raise ValueError(
                f"unsupported checkpoint version {version}; expected {CHECKPOINT_VERSION}"
            )
        return json.loads(str(archive["config_json"].item()))


def restore_resume_config(args, config):
    """Restore trajectory-defining CLI options from saved run metadata."""
    for name in RESUME_CONFIG_FIELDS:
        if name not in config:
            continue
        value = config[name]
        if name in {"channels", "blocks"}:
            value = tuple(value)
        setattr(args, name, value)

    # ``subset_size`` is only a spelling alias.  Checkpoints store the resolved
    # train limit so resuming does not depend on which spelling was used.
    train_limit = config.get("train_limit", config.get("subset_size"))
    args.train_limit = train_limit
    args.subset_size = None
    return args


def load_checkpoint(path, model, optimizer, strict=True, return_config=False):
    """Restore a complete checkpoint.

    The default two-item return value ``(epoch, history)`` preserves the old
    tutorial API.  Set ``return_config=True`` to receive
    ``(epoch, history, config)``.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        version = int(archive["checkpoint_version"].item())
        if version != CHECKPOINT_VERSION:
            raise ValueError(
                f"unsupported checkpoint version {version}; expected {CHECKPOINT_VERSION}"
            )
        model_state = {
            key.removeprefix("model::"): archive[key]
            for key in archive.files
            if key.startswith("model::")
        }
        load_model_state_dict(model, model_state, strict=strict)
        optimizer_arrays = {
            key.removeprefix("optimizer::"): archive[key]
            for key in archive.files
            if key.startswith("optimizer::")
        }
        optimizer_class = str(archive["optimizer_class"].item())
        optimizer_config = json.loads(str(archive["optimizer_config_json"].item()))
        optimizer_step = (
            int(archive["optimizer_step"].item())
            if "optimizer_step" in archive
            else None
        )
        _restore_optimizer(
            optimizer,
            optimizer_class,
            optimizer_config,
            optimizer_arrays,
            optimizer_step,
            strict=strict,
        )
        epoch = int(archive["epoch"].item())
        history = json.loads(str(archive["history_json"].item()))
        config = json.loads(str(archive["config_json"].item()))
        model_training = bool(archive["model_training"].item())
    if model_training:
        model.train()
    else:
        model.eval()
    if return_config:
        return epoch, history, config
    return epoch, history


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train NumPy SmallResNet on CIFAR-100.")
    parser.add_argument("--data-dir", type=Path, default=REPOSITORY_ROOT / "data")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--optimizer", choices=("momentum", "adamw"), default="momentum")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument(
        "--subset-size",
        type=int,
        default=None,
        help="alias for --train-limit; use --no-augment explicitly if desired",
    )
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPOSITORY_ROOT / "checkpoints" / "cifar100_numpy_resnet.npz",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "restore model/optimizer and trajectory-defining run config from "
            "--checkpoint; --epochs remains the new stopping epoch"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--channels", type=int, nargs=3, default=(16, 32, 64))
    parser.add_argument("--blocks", type=int, nargs=3, default=(2, 2, 2))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.resume and args.subset_size is not None and args.train_limit is not None:
        raise ValueError("use only one of --subset-size and --train-limit")
    if args.resume:
        saved_config = read_checkpoint_config(args.checkpoint)
        restore_resume_config(args, saved_config)
        print("restored model, optimizer, split, batch, seed, and augmentation config")
    train_limit = args.subset_size if args.subset_size is not None else args.train_limit
    use_augmentation = not args.no_augment
    np.random.seed(args.seed)
    train, validation, test = load_cifar100_splits(
        args.data_dir,
        val_size=args.val_size,
        seed=args.seed,
        train_limit=train_limit,
        val_limit=args.val_limit,
        test_limit=args.test_limit,
    )
    model = SmallResNet(
        num_classes=100,
        channels=tuple(args.channels),
        blocks_per_stage=tuple(args.blocks),
    )
    loss_fn = CrossEntropyLoss()
    optimizer = make_optimizer(
        args.optimizer, model.parameters(), args.lr, args.weight_decay
    )
    history = []
    start_epoch = 1
    if args.resume:
        last_epoch, history, _ = load_checkpoint(
            args.checkpoint, model, optimizer, return_config=True
        )
        start_epoch = last_epoch + 1
        print(f"resumed checkpoint={args.checkpoint} start_epoch={start_epoch}")

    run_config = vars(args).copy()
    run_config["train_limit"] = train_limit
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_accuracy = train_epoch(
            model,
            loss_fn,
            optimizer,
            *train,
            batch_size=args.batch_size,
            seed=args.seed + epoch,
            augment=use_augmentation,
        )
        val_loss, val_accuracy = evaluate(
            model, loss_fn, *validation, batch_size=args.batch_size
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_accuracy,
            "val_loss": val_loss,
            "val_acc": val_accuracy,
        }
        history.append(row)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} train_acc={train_accuracy:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.3f}"
        )
        save_checkpoint(
            args.checkpoint,
            model,
            optimizer,
            epoch=epoch,
            history=history,
            config=run_config,
        )

    test_loss, test_accuracy = evaluate(
        model, loss_fn, *test, batch_size=args.batch_size
    )
    print(f"test_loss={test_loss:.4f} test_acc={test_accuracy:.3f}")
    return model, history


if __name__ == "__main__":
    main()
