"""
Use the mini NumPy library from task_02 to train an MLP on MNIST.

This starter intentionally leaves data loading light. If torchvision is not
installed, install it or replace `load_mnist` with your own npz/csv loader.
"""

import sys
from pathlib import Path

import numpy as np

TASK_02 = Path(__file__).resolve().parents[1] / "task_02_mini_dl_lib"
sys.path.append(str(TASK_02))

from my_dl_lib import CrossEntropyLoss, Linear, Momentum, ReLU, Sequential  # noqa: E402


def one_hot(labels, num_classes=10):
    y = np.zeros((labels.shape[0], num_classes))
    y[np.arange(labels.shape[0]), labels] = 1
    return y


def load_mnist(sample_size=5000):
    from torchvision import datasets, transforms

    dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transforms.ToTensor()
    )
    x = dataset.data.numpy().reshape(-1, 784).astype("float32") / 255.0
    labels = dataset.targets.numpy()
    return x[:sample_size], one_hot(labels[:sample_size]), labels[:sample_size]


def main():
    x, y, labels = load_mnist()
    model = Sequential(Linear(784, 128), ReLU(), Linear(128, 10))
    loss_fn = CrossEntropyLoss()
    optimizer = Momentum(model.parameters(), lr=0.1)

    for epoch in range(20):
        logits = model.forward(x)
        loss = loss_fn.forward(logits, y)
        model.backward(loss_fn.backward())
        optimizer.step()
        acc = np.mean(np.argmax(logits, axis=1) == labels)
        print(f"epoch={epoch} loss={loss:.4f} train_acc={acc:.3f}")


if __name__ == "__main__":
    main()

