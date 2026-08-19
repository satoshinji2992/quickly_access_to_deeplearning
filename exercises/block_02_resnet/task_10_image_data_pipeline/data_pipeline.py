"""Data preparation helpers for CIFAR-100.

The official training split is divided *once* into stratified train and
validation subsets.  The official test split remains untouched until final
evaluation, preventing the common validation/test leakage mistake.
"""

import hashlib

import numpy as np


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def _validate_images_labels(images, labels, split_name="split"):
    images = np.asarray(images)
    labels = np.asarray(labels)
    if images.ndim < 2:
        raise ValueError(f"{split_name} images must have a leading sample dimension")
    if labels.ndim != 1:
        raise ValueError(f"{split_name} labels must be one-dimensional")
    if images.shape[0] != labels.shape[0]:
        raise ValueError(
            f"{split_name} has {images.shape[0]} images but {labels.shape[0]} labels"
        )
    if images.shape[0] == 0:
        raise ValueError(f"{split_name} must not be empty")
    return images, labels


def to_nchw(images):
    """Convert NHWC images to NCHW float32 images."""
    images = np.asarray(images)
    if images.ndim != 4:
        raise ValueError("expected images with shape (N, H, W, C)")
    return images.astype("float32", copy=False).transpose(0, 3, 1, 2)


def normalize(images, mean, std):
    """Scale 0..255 NCHW pixels and normalize each channel."""
    images = np.asarray(images)
    if images.ndim != 4:
        raise ValueError("expected images with shape (N, C, H, W)")
    channels = images.shape[1]
    mean = np.asarray(mean, dtype="float32")
    std = np.asarray(std, dtype="float32")
    if mean.shape != (channels,) or std.shape != (channels,):
        raise ValueError(f"mean and std must each contain {channels} channel values")
    if np.any(std <= 0) or not np.all(np.isfinite(std)):
        raise ValueError("std values must be positive and finite")
    mean = mean.reshape(1, channels, 1, 1)
    std = std.reshape(1, channels, 1, 1)
    return (images.astype("float32", copy=False) / 255.0 - mean) / std


def random_horizontal_flip(images, p=0.5, rng=None):
    """Independently flip each NCHW image along its width axis."""
    images = np.asarray(images)
    if images.ndim != 4:
        raise ValueError("expected images with shape (N, C, H, W)")
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")
    rng = np.random.default_rng() if rng is None else rng
    mask = rng.random(images.shape[0]) < p
    output = images.copy()
    output[mask] = output[mask, :, :, ::-1]
    return output


def random_crop_with_padding(images, padding=4, crop_size=32, rng=None):
    """Zero-pad NCHW images, then sample one crop per image."""
    images = np.asarray(images)
    if images.ndim != 4:
        raise ValueError("expected images with shape (N, C, H, W)")
    padding = int(padding)
    crop_size = int(crop_size)
    if padding < 0 or crop_size <= 0:
        raise ValueError("padding must be non-negative and crop_size positive")
    rng = np.random.default_rng() if rng is None else rng
    sample_count, channels, height, width = images.shape
    padded_height = height + 2 * padding
    padded_width = width + 2 * padding
    if crop_size > padded_height or crop_size > padded_width:
        raise ValueError("crop_size is larger than the padded image")
    padded = np.pad(
        images,
        ((0, 0), (0, 0), (padding, padding), (padding, padding)),
        mode="constant",
    )
    output = np.empty(
        (sample_count, channels, crop_size, crop_size), dtype=images.dtype
    )
    for sample in range(sample_count):
        top = rng.integers(0, padded_height - crop_size + 1)
        left = rng.integers(0, padded_width - crop_size + 1)
        output[sample] = padded[
            sample, :, top : top + crop_size, left : left + crop_size
        ]
    return output


def iterate_minibatches(images, labels, batch_size, shuffle=True, seed=None):
    """Yield aligned image/label mini-batches, including the final short batch."""
    images, labels = _validate_images_labels(images, labels)
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    rng = np.random.default_rng(seed)
    indices = np.arange(images.shape[0])
    if shuffle:
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        yield images[batch_indices], labels[batch_indices]


def _allocate_per_class(class_counts, total, per_class_maximum):
    """Allocate an exact total approximately proportional to class counts."""
    class_counts = np.asarray(class_counts, dtype=np.int64)
    maxima = np.asarray(per_class_maximum, dtype=np.int64)
    total = int(total)
    if total < 0 or total > int(np.sum(maxima)):
        raise ValueError(
            f"requested subset size {total} is outside [0, {int(np.sum(maxima))}]"
        )
    if total == 0:
        return np.zeros_like(class_counts)
    ideal = total * class_counts / np.sum(class_counts)
    allocation = np.minimum(np.floor(ideal).astype(np.int64), maxima)
    remaining = total - int(np.sum(allocation))
    # Largest-remainder allocation, repeated when a class was clipped.
    priority = np.argsort(-(ideal - np.floor(ideal)), kind="stable")
    while remaining:
        made_progress = False
        for index in priority:
            if allocation[index] < maxima[index]:
                allocation[index] += 1
                remaining -= 1
                made_progress = True
                if remaining == 0:
                    break
        if not made_progress:  # Defensive; the range check above should prevent this.
            raise RuntimeError("could not allocate the requested stratified subset")
    return allocation


def stratified_train_val_split(
    images,
    labels,
    val_size=None,
    val_fraction=0.1,
    seed=0,
    return_indices=False,
):
    """Split samples into independent, approximately stratified train/val sets.

    Every class with at least two samples keeps at least one training example.
    ``val_size`` takes precedence over ``val_fraction``.
    """
    images, labels = _validate_images_labels(images, labels, "full training split")
    sample_count = labels.shape[0]
    if val_size is None:
        if not 0.0 < val_fraction < 1.0:
            raise ValueError("val_fraction must be in (0, 1)")
        val_size = int(round(sample_count * val_fraction))
    val_size = int(val_size)
    if val_size <= 0 or val_size >= sample_count:
        raise ValueError("val_size must leave both train and validation non-empty")

    classes, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    del classes
    maximum_val_counts = np.maximum(counts - 1, 0)
    val_counts = _allocate_per_class(counts, val_size, maximum_val_counts)
    rng = np.random.default_rng(seed)
    train_indices = []
    val_indices = []
    for class_index, class_val_count in enumerate(val_counts):
        indices = np.flatnonzero(inverse == class_index)
        rng.shuffle(indices)
        val_indices.extend(indices[:class_val_count])
        train_indices.extend(indices[class_val_count:])
    train_indices = np.asarray(train_indices, dtype=np.int64)
    val_indices = np.asarray(val_indices, dtype=np.int64)
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    if np.intersect1d(train_indices, val_indices).size:
        raise RuntimeError("internal error: train and validation indices overlap")

    train = (images[train_indices].copy(), labels[train_indices].copy())
    validation = (images[val_indices].copy(), labels[val_indices].copy())
    if return_indices:
        return train, validation, (train_indices, val_indices)
    return train, validation


def stratified_subset(images, labels, size, seed=0):
    """Return a deterministic stratified subset without mutating the inputs."""
    images, labels = _validate_images_labels(images, labels)
    size = int(size)
    if size <= 0 or size > labels.shape[0]:
        raise ValueError("size must be in [1, number of samples]")
    if size == labels.shape[0]:
        return images.copy(), labels.copy()
    _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    allocations = _allocate_per_class(counts, size, counts)
    rng = np.random.default_rng(seed)
    selected = []
    for class_index, allocation in enumerate(allocations):
        candidates = np.flatnonzero(inverse == class_index)
        rng.shuffle(candidates)
        selected.extend(candidates[:allocation])
    selected = np.asarray(selected, dtype=np.int64)
    rng.shuffle(selected)
    return images[selected].copy(), labels[selected].copy()


def _sample_fingerprints(images):
    images = np.asarray(images)
    fingerprints = set()
    for sample in images:
        contiguous = np.ascontiguousarray(sample)
        digest = hashlib.blake2b(digest_size=16)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(contiguous.tobytes())
        fingerprints.add(digest.digest())
    return fingerprints


def assert_disjoint_splits(*splits):
    """Raise ``AssertionError`` when exact image records occur across splits.

    Each argument may be an image array or an ``(images, labels)`` pair.  This
    content-based check is useful for generated/local datasets where source
    indices are not available.
    """
    fingerprints = []
    for split in splits:
        images = split[0] if isinstance(split, tuple) else split
        fingerprints.append(_sample_fingerprints(images))
    for left in range(len(fingerprints)):
        for right in range(left + 1, len(fingerprints)):
            overlap = fingerprints[left].intersection(fingerprints[right])
            if overlap:
                raise AssertionError(
                    f"data leakage: splits {left} and {right} share "
                    f"{len(overlap)} exact image(s)"
                )


def validate_splits(train, validation, test, num_classes=100, check_content=True):
    """Validate shapes, label ranges, and optionally exact cross-split overlap."""
    for name, split in zip(("train", "validation", "test"), (train, validation, test)):
        images, labels = _validate_images_labels(*split, split_name=name)
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError(f"{name} labels must be integers")
        if np.any(labels < 0) or np.any(labels >= num_classes):
            raise ValueError(f"{name} labels must be in [0, {num_classes})")
    if check_content:
        assert_disjoint_splits(train, validation, test)
    return True


def load_cifar100_splits(
    data_dir,
    val_size=5000,
    seed=0,
    download=True,
    train_limit=None,
    val_limit=None,
    test_limit=None,
    normalized=True,
):
    """Load isolated CIFAR-100 train/validation/test splits.

    Validation is sampled from the official 50,000-image training set.  Test
    is always sampled from the separate official 10,000-image test set and is
    therefore never used for model selection.
    """
    try:
        from torchvision.datasets import CIFAR100
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "torchvision is required for CIFAR-100; install requirements.txt"
        ) from exc

    official_train = CIFAR100(root=str(data_dir), train=True, download=download)
    official_test = CIFAR100(root=str(data_dir), train=False, download=download)
    full_train_images = np.asarray(official_train.data)
    full_train_labels = np.asarray(official_train.targets, dtype=np.int64)
    test_images = np.asarray(official_test.data).copy()
    test_labels = np.asarray(official_test.targets, dtype=np.int64).copy()

    train, validation, indices = stratified_train_val_split(
        full_train_images,
        full_train_labels,
        val_size=val_size,
        seed=seed,
        return_indices=True,
    )
    train_indices, validation_indices = indices
    if np.intersect1d(train_indices, validation_indices).size:
        raise RuntimeError("train/validation leakage detected")

    if train_limit is not None:
        train = stratified_subset(*train, size=train_limit, seed=seed + 1)
    if val_limit is not None:
        validation = stratified_subset(*validation, size=val_limit, seed=seed + 2)
    test = (test_images, test_labels)
    if test_limit is not None:
        test = stratified_subset(*test, size=test_limit, seed=seed + 3)

    def prepare(split):
        split_images, split_labels = split
        split_images = to_nchw(split_images)
        if normalized:
            split_images = normalize(split_images, CIFAR100_MEAN, CIFAR100_STD)
        return split_images, split_labels

    # The official train and test datasets have distinct source identities;
    # train/validation non-overlap was checked using their source indices above.
    return prepare(train), prepare(validation), prepare(test)
