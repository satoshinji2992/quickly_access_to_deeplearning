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
        raise NotImplementedError("implement Linear.forward")

    def backward(self, dout):
        raise NotImplementedError("implement Linear.backward")

    def parameters(self):
        return [(self.W, self.dW), (self.b, self.db)]


class ReLU:
    def forward(self, x):
        raise NotImplementedError("implement ReLU.forward")

    def backward(self, dout):
        raise NotImplementedError("implement ReLU.backward")


class GELU:
    def forward(self, x):
        raise NotImplementedError("implement GELU.forward")

    def backward(self, dout):
        raise NotImplementedError("implement GELU.backward")


class CrossEntropyLoss:
    def forward(self, logits, targets):
        raise NotImplementedError("implement CrossEntropyLoss.forward")

    def backward(self):
        raise NotImplementedError("implement CrossEntropyLoss.backward")


class Sequential:
    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x):
        raise NotImplementedError("implement Sequential.forward")

    def backward(self, dout):
        raise NotImplementedError("implement Sequential.backward")

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
        raise NotImplementedError("implement SGD.step")


class Momentum:
    def __init__(self, parameters, lr=0.01, beta=0.9):
        self.parameters = list(parameters)
        self.lr = lr
        self.beta = beta
        self.velocity = [np.zeros_like(value) for value, _ in self.parameters]

    def step(self):
        raise NotImplementedError("implement Momentum.step")
