import numpy as np


class Linear:
    def __init__(self, in_features, out_features):
        scale = np.sqrt(2.0 / in_features)
        self.W = np.random.randn(in_features, out_features) * scale
        self.b = np.zeros((1, out_features))
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.x = None

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout):
        self.dW[...] = self.x.T @ dout
        self.db[...] = np.sum(dout, axis=0, keepdims=True)
        return dout @ self.W.T

    def parameters(self):
        return [(self.W, self.dW), (self.b, self.db)]


class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = x > 0
        return np.maximum(0, x)

    def backward(self, dout):
        return dout * self.mask


class GELU:
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        c = np.sqrt(2.0 / np.pi)
        return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x**3)))

    def backward(self, dout):
        x = self.x
        c = np.sqrt(2.0 / np.pi)
        u = c * (x + 0.044715 * x**3)
        tanh_u = np.tanh(u)
        du = c * (1.0 + 3.0 * 0.044715 * x**2)
        grad = 0.5 * (1.0 + tanh_u) + 0.5 * x * (1.0 - tanh_u**2) * du
        return dout * grad


class CrossEntropyLoss:
    def __init__(self):
        self.probs = None
        self.targets = None

    def forward(self, logits, targets):
        self.targets = targets
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(shifted)
        self.probs = exp / np.sum(exp, axis=1, keepdims=True)
        eps = 1e-12
        return -np.mean(np.sum(targets * np.log(self.probs + eps), axis=1))

    def backward(self):
        return (self.probs - self.targets) / self.targets.shape[0]


class Sequential:
    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def parameters(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, "parameters"):
                params.extend(layer.parameters())
        return params


class SGD:
    def __init__(self, parameters, lr=0.01):
        self.parameters = list(parameters)
        self.lr = lr

    def step(self):
        for value, grad in self.parameters:
            value -= self.lr * grad


class Momentum:
    def __init__(self, parameters, lr=0.01, beta=0.9):
        self.parameters = list(parameters)
        self.lr = lr
        self.beta = beta
        self.velocity = [np.zeros_like(value) for value, _ in self.parameters]

    def step(self):
        for idx, (value, grad) in enumerate(self.parameters):
            self.velocity[idx] = self.beta * self.velocity[idx] + (1 - self.beta) * grad
            value -= self.lr * self.velocity[idx]
