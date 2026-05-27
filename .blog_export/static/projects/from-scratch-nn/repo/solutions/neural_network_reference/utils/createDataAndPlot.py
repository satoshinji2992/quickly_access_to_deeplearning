# generate data
import numpy as np
import random
import matplotlib.pyplot as plt

NUM_OF_DATA = 10000


def tag_entry_circle(x, y):
    if x**2 + y**2 < 1:
        tag = 0
    else:
        tag = 1
    return tag


def tag_entry_triangle(x, y):
    x = x + 1
    y = y + 1
    if (
        (0 <= x <= 1.5)
        and (0 <= y <= 2)
        and (y >= (2 / 1.5) * x and y >= -(2 / 1.5) * (x - 1.5))
    ):
        tag = 0
    else:
        tag = 1
    return tag


def tag_entry_square(x, y):
    x = x + 1
    y = y + 1
    if (0 <= x <= 2) and (0 <= y <= 2):
        tag = 0
    else:
        tag = 1
    return tag


function = {
    "circle": tag_entry_circle,
    "triangle": tag_entry_triangle,
    "square": tag_entry_square,
}


def create_data(num_of_data=NUM_OF_DATA, mode="circle"):
    data = []
    for i in range(num_of_data):
        x = random.uniform(-2, 2)
        y = random.uniform(-2, 2)
        tag = function[mode](x, y)
        entry = [x, y, tag]
        data.append(entry)
    return np.array(data)


def plot_data(data1, data2, title):
    fig, ax = plt.subplots(1, 2)
    ax[0].scatter(data1[:, 0], data1[:, 1], c=data1[:, 2], cmap=plt.cm.coolwarm)
    ax[0].set_title(title + "1")
    ax[1].scatter(data2[:, 0], data2[:, 1], c=data2[:, 2], cmap=plt.cm.coolwarm)
    ax[1].set_title(title + "2")
    plt.show()

if __name__ == "__main__":
    data = create_data(1024, "circle")
    fig, ax = plt.subplots(1, 1)
    ax.scatter(data[:, 0], data[:, 1], c=data[:, 2], cmap=plt.cm.coolwarm)
    plt.savefig('data_circle.png')
    plt.show()
    plt.close()