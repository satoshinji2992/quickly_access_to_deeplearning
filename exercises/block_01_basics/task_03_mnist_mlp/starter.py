"""Use the mini NumPy library from task_02 to train an MLP on MNIST."""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT))

from common.my_dl_lib import CrossEntropyLoss, Linear, Momentum, ReLU, Sequential  # noqa: E402


def one_hot(labels, num_classes=10):
    labels = np.asarray(labels, dtype=np.int64)
    y = np.zeros((labels.shape[0], num_classes))
    y[np.arange(labels.shape[0]), labels] = 1
    return y


def stratified_train_val_split(x, labels, train_size, val_size, seed=42):
    """Return disjoint train/validation arrays with preserved class ratios."""

    from sklearn.model_selection import train_test_split

    x = np.asarray(x)
    labels = np.asarray(labels, dtype=np.int64)
    if len(x) != len(labels):
        raise ValueError("x and labels must contain the same number of examples")
    if train_size <= 0 or val_size <= 0 or train_size + val_size > len(x):
        raise ValueError("train_size and val_size must be positive and fit in the dataset")

    indices = np.arange(len(labels))
    selected_size = train_size + val_size
    if selected_size < len(indices):
        selected, _ = train_test_split(
            indices,
            train_size=selected_size,
            random_state=seed,
            stratify=labels,
        )
    else:
        selected = indices

    train_indices, val_indices = train_test_split(
        selected,
        train_size=train_size,
        test_size=val_size,
        random_state=seed,
        stratify=labels[selected],
    )
    return x[train_indices], labels[train_indices], x[val_indices], labels[val_indices]


def _fit_split_sizes(dataset_size, train_size, val_size):
    """Keep the requested ratio when a fallback dataset is smaller than MNIST."""

    if train_size + val_size <= dataset_size:
        return train_size, val_size
    val_fraction = val_size / (train_size + val_size)
    fitted_val_size = max(1, int(round(dataset_size * val_fraction)))
    return dataset_size - fitted_val_size, fitted_val_size


def load_mnist(train_size=5000, val_size=1000, seed=42):
    """Load MNIST and make a real held-out, stratified validation split."""

    try:
        from torchvision import datasets

        data_root = Path(__file__).resolve().parent / "data"
        dataset = datasets.MNIST(root=str(data_root), train=True, download=True)
        x = dataset.data.numpy().reshape(-1, 784).astype("float32") / 255.0
        labels = dataset.targets.numpy().astype("int64")
    except (ModuleNotFoundError, ImportError):
        from sklearn.datasets import load_digits

        print("torchvision is not installed; using sklearn digits as a small fallback.")
        digits = load_digits()
        images = digits.images.astype("float32") / 16.0
        padded = np.zeros((images.shape[0], 28, 28), dtype="float32")
        padded[:, 10:18, 10:18] = images
        labels = digits.target.astype("int64")
        x = padded.reshape(-1, 784)

    train_size, val_size = _fit_split_sizes(len(x), train_size, val_size)
    x_train, train_labels, x_val, val_labels = stratified_train_val_split(
        x, labels, train_size=train_size, val_size=val_size, seed=seed
    )
    return (
        x_train,
        one_hot(train_labels),
        train_labels,
        x_val,
        one_hot(val_labels),
        val_labels,
    )


def evaluate(model, loss_fn, x, targets, labels):
    logits = model.forward(x)
    loss = loss_fn.forward(logits, targets)
    accuracy = float(np.mean(np.argmax(logits, axis=1) == labels))
    return loss, accuracy


def main():
    x_train, y_train, train_labels, x_val, y_val, val_labels = load_mnist()
    print(f"data split: train={len(x_train)}, val={len(x_val)} (stratified, disjoint)")

    model = Sequential(Linear(784, 128), ReLU(), Linear(128, 10))
    loss_fn = CrossEntropyLoss()
    optimizer = Momentum(model.parameters(), lr=0.1)

    for epoch in range(20):
        model.train()
        logits = model.forward(x_train)
        loss_fn.forward(logits, y_train)
        model.backward(loss_fn.backward())
        optimizer.step()

        # Metrics are recomputed after the update. Validation data never enters
        # backward(), so val_acc measures behavior on unseen examples.
        model.eval()
        train_loss, train_acc = evaluate(
            model, loss_fn, x_train, y_train, train_labels
        )
        val_loss, val_acc = evaluate(model, loss_fn, x_val, y_val, val_labels)
        print(
            f"epoch={epoch + 1:02d} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )


if __name__ == "__main__":
    main()
