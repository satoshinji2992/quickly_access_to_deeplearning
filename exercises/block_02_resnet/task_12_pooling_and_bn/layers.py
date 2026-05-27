import numpy as np


class MaxPool2D:
    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        raise NotImplementedError("implement max pooling forward")

    def backward(self, dout):
        raise NotImplementedError("implement max pooling backward")


class GlobalAvgPool2D:
    def forward(self, x):
        self.x_shape = x.shape
        return np.mean(x, axis=(2, 3))

    def backward(self, dout):
        n, c, h, w = self.x_shape
        return dout[:, :, None, None] * np.ones((n, c, h, w)) / (h * w)


class BatchNorm2D:
    def __init__(self, channels, momentum=0.9, eps=1e-5):
        self.gamma = np.ones((1, channels, 1, 1))
        self.beta = np.zeros((1, channels, 1, 1))
        self.running_mean = np.zeros((1, channels, 1, 1))
        self.running_var = np.ones((1, channels, 1, 1))
        self.momentum = momentum
        self.eps = eps
        self.training = True

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def forward(self, x):
        if self.training:
            mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
            var = np.var(x, axis=(0, 2, 3), keepdims=True)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            mean = self.running_mean
            var = self.running_var
        self.x_centered = x - mean
        self.std_inv = 1.0 / np.sqrt(var + self.eps)
        self.x_hat = self.x_centered * self.std_inv
        return self.gamma * self.x_hat + self.beta

    def backward(self, dout):
        raise NotImplementedError("implement BatchNorm2D backward")

