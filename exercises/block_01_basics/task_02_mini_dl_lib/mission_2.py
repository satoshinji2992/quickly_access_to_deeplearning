import numpy as np

from my_dl_lib import CrossEntropyLoss, Linear, Momentum, ReLU, Sequential


def make_circle_data(n=2000, radius=1.0, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.5, 1.5, size=(n, 2))
    label = (np.sum(x**2, axis=1) <= radius**2).astype(int)
    y = np.zeros((n, 2))
    y[np.arange(n), label] = 1
    return x, y, label


def accuracy(logits, labels):
    return np.mean(np.argmax(logits, axis=1) == labels)


def main():
    x, y, labels = make_circle_data()
    model = Sequential(Linear(2, 16), ReLU(), Linear(16, 16), ReLU(), Linear(16, 2))
    loss_fn = CrossEntropyLoss()
    optimizer = Momentum(model.parameters(), lr=0.2)

    for epoch in range(500):
        logits = model.forward(x)
        loss = loss_fn.forward(logits, y)
        model.backward(loss_fn.backward())
        optimizer.step()
        if epoch % 50 == 0:
            print(f"epoch={epoch} loss={loss:.4f} acc={accuracy(logits, labels):.3f}")


if __name__ == "__main__":
    main()

