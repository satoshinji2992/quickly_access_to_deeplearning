import numpy as np


def to_nchw(images):
    """Convert NHWC images to NCHW float32 images."""
    images = images.astype("float32")
    if images.ndim != 4:
        raise ValueError("expected images with shape (N, H, W, C)")
    return images.transpose(0, 3, 1, 2)


def normalize(images, mean, std):
    mean = np.asarray(mean, dtype="float32").reshape(1, 3, 1, 1)
    std = np.asarray(std, dtype="float32").reshape(1, 3, 1, 1)
    return (images / 255.0 - mean) / std


def random_horizontal_flip(images, p=0.5, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    mask = rng.random(images.shape[0]) < p
    images = images.copy()
    images[mask] = images[mask, :, :, ::-1]
    return images


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


def iterate_minibatches(images, labels, batch_size, shuffle=True, seed=None):
    rng = np.random.default_rng(seed)
    indices = np.arange(images.shape[0])
    if shuffle:
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        yield images[batch_idx], labels[batch_idx]

