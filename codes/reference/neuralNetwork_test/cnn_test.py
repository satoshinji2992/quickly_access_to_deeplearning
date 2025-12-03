import numpy as np
import matplotlib.pyplot as plt
import network_test as nt
import os
from torchvision import datasets, transforms


def download_mnist(data_dir="./data/MNIST/raw"):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transforms.ToTensor()
    )

    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transforms.ToTensor()
    )

    train_images = train_dataset.data.numpy().reshape(-1, 28 * 28)
    train_labels = train_dataset.targets.numpy()

    test_images = test_dataset.data.numpy().reshape(-1, 28 * 28)
    test_labels = test_dataset.targets.numpy()

    return train_images, train_labels, test_images, test_labels


def preprocess_data(images, labels):

    images = images / 255.0

    one_hot = np.zeros((labels.size, 10))
    one_hot[np.arange(labels.size), labels] = 1

    return images, one_hot


def main():
    train_images, train_labels, test_images, test_labels = download_mnist()

    train_images, train_one_hot = preprocess_data(train_images, train_labels)
    test_images, test_one_hot = preprocess_data(test_images, test_labels)

    sample_size = 5000
    train_images_sample = train_images[:sample_size]
    train_one_hot_sample = train_one_hot[:sample_size]

    network_shape = [784, 100, 10]

    nt.BACKPROPAGATION = "adam"  # 使用Adam优化
    nt.BATCH_SIZE = 64  # 设置批量大小

    network = nt.Network(network_shape, nt.activation_softmax)

    network.train(
        train_images_sample, train_one_hot_sample, learning_rate=0.01, epochs=100
    )

    test_sample_size = 1000
    test_outputs = network.network_forward(test_images[:test_sample_size])
    predictions = np.argmax(test_outputs[-1], axis=1)
    actual = np.argmax(test_one_hot[:test_sample_size], axis=1)

    accuracy = np.mean(predictions == actual)
    print(f"acurate: {accuracy * 100:.2f}%")

    plt.figure(figsize=(12, 6))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(test_images[i].reshape(28, 28), cmap="gray")
        plt.title(f"预测: {predictions[i]}, 实际: {actual[i]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
