"""Generate and validate the data used by the circle-classification task."""

from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CONDITION = "(x**2 + y**2) <= 1.0**2"


def labels_from_condition(x, y, condition=DEFAULT_CONDITION):
    """Evaluate a vectorized teaching condition and return binary labels."""

    x = np.asarray(x)
    y = np.asarray(y)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    namespace = {"x": x, "y": y, "np": np, "abs": np.abs}
    result = np.asarray(eval(condition, {"__builtins__": {}}, namespace), dtype=bool)
    if result.ndim == 0:
        result = np.full(x.shape, result, dtype=bool)
    if result.shape != x.shape:
        raise ValueError(
            f"condition must return one label per point: expected {x.shape}, got {result.shape}"
        )
    return result.astype(np.int64)


def create_data(
    n=1000,
    variance=1.0,
    out_path="data.csv",
    condition=DEFAULT_CONDITION,
    seed=42,
):
    """Create one labelled sample file.

    Prefer :func:`create_data_splits` for model training: it makes the train and
    validation files together, so their separation and label proportions can be
    checked as one operation.
    """

    if n <= 0:
        raise ValueError("n must be positive")
    if variance <= 0:
        raise ValueError("variance must be positive")

    rng = np.random.default_rng(seed)
    samples = rng.normal(loc=0.0, scale=np.sqrt(variance), size=(n, 2))
    labels = labels_from_condition(samples[:, 0], samples[:, 1], condition)
    dataframe = pd.DataFrame(
        {"x": samples[:, 0], "y": samples[:, 1], "label": labels}
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(out_path, index=False)
    return dataframe


def _stratified_partition(dataframe, val_n, rng):
    """Partition a dataframe while preserving class proportions."""

    labels = dataframe["label"].to_numpy(dtype=np.int64)
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) < 2 or np.any(counts < 2):
        raise ValueError(
            "the generated pool needs at least two points from each class; "
            "adjust the condition or coordinate variance"
        )
    if val_n < len(classes) or len(dataframe) - val_n < len(classes):
        raise ValueError("train_n and val_n must each be large enough to contain every class")

    desired = counts * (val_n / len(dataframe))
    val_counts = np.floor(desired).astype(int)
    val_counts = np.clip(val_counts, 1, counts - 1)

    while val_counts.sum() < val_n:
        candidates = np.flatnonzero(val_counts < counts - 1)
        if candidates.size == 0:
            raise ValueError("cannot allocate the requested validation split")
        best = candidates[np.argmax(desired[candidates] - val_counts[candidates])]
        val_counts[best] += 1
    while val_counts.sum() > val_n:
        candidates = np.flatnonzero(val_counts > 1)
        if candidates.size == 0:
            raise ValueError("cannot allocate the requested validation split")
        best = candidates[np.argmax(val_counts[candidates] - desired[candidates])]
        val_counts[best] -= 1

    train_indices = []
    val_indices = []
    for label, class_val_n in zip(classes, val_counts):
        class_indices = np.flatnonzero(labels == label)
        rng.shuffle(class_indices)
        val_indices.extend(class_indices[:class_val_n])
        train_indices.extend(class_indices[class_val_n:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    train = dataframe.iloc[train_indices].reset_index(drop=True)
    val = dataframe.iloc[val_indices].reset_index(drop=True)
    return train, val


def validate_data_splits(train, val, condition=DEFAULT_CONDITION):
    """Check labels, class stratification and train/validation separation.

    ``train`` and ``val`` may be dataframes or CSV paths. A ``ValueError`` is
    raised immediately if a stale or leaked file is encountered.
    """

    if not isinstance(train, pd.DataFrame):
        train = pd.read_csv(train)
    if not isinstance(val, pd.DataFrame):
        val = pd.read_csv(val)

    required_columns = {"x", "y", "label"}
    for split_name, dataframe in (("train", train), ("val", val)):
        if not required_columns.issubset(dataframe.columns):
            raise ValueError(f"{split_name} data must contain x, y and label columns")
        values = dataframe[["x", "y"]].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{split_name} data contains a non-finite coordinate")
        labels = dataframe["label"].to_numpy()
        if not np.isin(labels, (0, 1)).all():
            raise ValueError(f"{split_name} labels must be binary (0 or 1)")
        expected = labels_from_condition(values[:, 0], values[:, 1], condition)
        if not np.array_equal(labels.astype(np.int64), expected):
            mismatch_count = int(np.count_nonzero(labels.astype(np.int64) != expected))
            raise ValueError(
                f"{split_name} has {mismatch_count} labels that do not match {condition!r}"
            )
        if dataframe.duplicated(subset=["x", "y"]).any():
            raise ValueError(f"{split_name} contains duplicate points")

    train_points = set(map(tuple, train[["x", "y"]].to_numpy()))
    val_points = set(map(tuple, val[["x", "y"]].to_numpy()))
    overlap = train_points.intersection(val_points)
    if overlap:
        raise ValueError(f"train and val overlap on {len(overlap)} points")

    train_counts = train["label"].value_counts(normalize=True)
    val_counts = val["label"].value_counts(normalize=True)
    if set(train_counts.index) != {0, 1} or set(val_counts.index) != {0, 1}:
        raise ValueError("both train and val must contain examples from both classes")
    tolerance = max(1.0 / len(train), 1.0 / len(val)) + 1e-12
    if any(abs(train_counts[label] - val_counts[label]) > tolerance for label in (0, 1)):
        raise ValueError("train and val class proportions are not stratified")

    return True


def create_data_splits(
    train_n=800,
    val_n=200,
    train_out_path="train_data.csv",
    val_out_path="val_data.csv",
    condition=DEFAULT_CONDITION,
    variance=1.0,
    seed=42,
):
    """Generate one pool and make deterministic, disjoint stratified splits."""

    if train_n <= 0 or val_n <= 0:
        raise ValueError("train_n and val_n must be positive")
    if variance <= 0:
        raise ValueError("variance must be positive")

    rng = np.random.default_rng(seed)
    samples = rng.normal(
        loc=0.0, scale=np.sqrt(variance), size=(train_n + val_n, 2)
    )
    dataframe = pd.DataFrame(
        {
            "x": samples[:, 0],
            "y": samples[:, 1],
            "label": labels_from_condition(samples[:, 0], samples[:, 1], condition),
        }
    )
    train, val = _stratified_partition(dataframe, val_n, rng)
    if len(train) != train_n or len(val) != val_n:
        raise RuntimeError("internal split size error")
    validate_data_splits(train, val, condition)

    train_out_path = Path(train_out_path)
    val_out_path = Path(val_out_path)
    train_out_path.parent.mkdir(parents=True, exist_ok=True)
    val_out_path.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(train_out_path, index=False)
    val.to_csv(val_out_path, index=False)
    return train, val


if __name__ == "__main__":
    task_dir = Path(__file__).resolve().parent
    train_data, val_data = create_data_splits(
        train_out_path=task_dir / "train_data.csv",
        val_out_path=task_dir / "val_data.csv",
    )
    print(
        "generated valid splits:",
        f"train={len(train_data)}, val={len(val_data)},",
        f"positive_ratio={train_data['label'].mean():.3f}/{val_data['label'].mean():.3f}",
    )
