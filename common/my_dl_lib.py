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
        return np.dot(x, self.W) + self.b

    def backward(self, dout):
        self.dW[...] = np.dot(self.x.T, dout)
        self.db[...] = np.sum(dout, axis=0, keepdims=True)
        return np.dot(dout, self.W.T)

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


class SiLU:
    def __init__(self):
        self.x = None
        self.sigmoid = None

    def forward(self, x):
        self.x = x
        self.sigmoid = 1.0 / (1.0 + np.exp(-x))
        return x * self.sigmoid

    def backward(self, dout):
        grad = self.sigmoid + self.x * self.sigmoid * (1.0 - self.sigmoid)
        return dout * grad


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


class BatchNorm1D:
    def __init__(self, features, momentum=0.9, eps=1e-5):
        self.gamma = np.ones((1, features))
        self.beta = np.zeros((1, features))
        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta = np.zeros_like(self.beta)
        self.running_mean = np.zeros((1, features))
        self.running_var = np.ones((1, features))
        self.momentum = momentum
        self.eps = eps
        self.training = True

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def forward(self, x):
        if self.training:
            mean = np.mean(x, axis=0, keepdims=True)
            var = np.var(x, axis=0, keepdims=True)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            mean = self.running_mean
            var = self.running_var
        self.x_hat = (x - mean) / np.sqrt(var + self.eps)
        self.std_inv = 1.0 / np.sqrt(var + self.eps)
        return self.gamma * self.x_hat + self.beta

    def backward(self, dout):
        n = dout.shape[0]
        self.dbeta[...] = np.sum(dout, axis=0, keepdims=True)
        self.dgamma[...] = np.sum(dout * self.x_hat, axis=0, keepdims=True)
        dxhat = dout * self.gamma
        sum_dxhat = np.sum(dxhat, axis=0, keepdims=True)
        sum_dxhat_xhat = np.sum(dxhat * self.x_hat, axis=0, keepdims=True)
        return (dxhat - sum_dxhat / n - self.x_hat * sum_dxhat_xhat / n) * self.std_inv

    def parameters(self):
        return [(self.gamma, self.dgamma), (self.beta, self.dbeta)]


class LayerNorm:
    def __init__(self, features, eps=1e-5):
        self.gamma = np.ones((1, features))
        self.beta = np.zeros((1, features))
        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta = np.zeros_like(self.beta)
        self.eps = eps

    def forward(self, x):
        self.mean = np.mean(x, axis=-1, keepdims=True)
        self.var = np.var(x, axis=-1, keepdims=True)
        self.std_inv = 1.0 / np.sqrt(self.var + self.eps)
        self.x_hat = (x - self.mean) * self.std_inv
        return self.gamma * self.x_hat + self.beta

    def backward(self, dout):
        feature_count = dout.shape[-1]
        self.dbeta[...] = np.sum(dout, axis=0, keepdims=True)
        self.dgamma[...] = np.sum(dout * self.x_hat, axis=0, keepdims=True)
        dxhat = dout * self.gamma
        sum_dxhat = np.sum(dxhat, axis=-1, keepdims=True)
        sum_dxhat_xhat = np.sum(dxhat * self.x_hat, axis=-1, keepdims=True)
        return (
            dxhat - sum_dxhat / feature_count - self.x_hat * sum_dxhat_xhat / feature_count
        ) * self.std_inv

    def parameters(self):
        return [(self.gamma, self.dgamma), (self.beta, self.dbeta)]


class Dropout:
    def __init__(self, p=0.5, seed=None):
        if not 0 <= p < 1:
            raise ValueError("p must be in [0, 1)")
        self.p = p
        self.rng = np.random.default_rng(seed)
        self.training = True
        self.mask = None

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def forward(self, x):
        if not self.training or self.p == 0:
            self.mask = None
            return x
        keep_prob = 1.0 - self.p
        self.mask = (self.rng.random(x.shape) < keep_prob) / keep_prob
        return x * self.mask

    def backward(self, dout):
        if self.mask is None:
            return dout
        return dout * self.mask


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

    def train(self):
        for layer in self.layers:
            if hasattr(layer, "train"):
                layer.train()

    def eval(self):
        for layer in self.layers:
            if hasattr(layer, "eval"):
                layer.eval()


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


class Adagrad:
    def __init__(self, parameters, lr=0.01, eps=1e-8):
        self.parameters = list(parameters)
        self.lr = lr
        self.eps = eps
        self.cache = [np.zeros_like(value) for value, _ in self.parameters]

    def step(self):
        for idx, (value, grad) in enumerate(self.parameters):
            self.cache[idx] += grad**2
            value -= self.lr * grad / (np.sqrt(self.cache[idx]) + self.eps)


class RMSProp:
    def __init__(self, parameters, lr=0.001, rho=0.9, eps=1e-8):
        self.parameters = list(parameters)
        self.lr = lr
        self.rho = rho
        self.eps = eps
        self.cache = [np.zeros_like(value) for value, _ in self.parameters]

    def step(self):
        for idx, (value, grad) in enumerate(self.parameters):
            self.cache[idx] = self.rho * self.cache[idx] + (1 - self.rho) * grad**2
            value -= self.lr * grad / (np.sqrt(self.cache[idx]) + self.eps)


class Adam:
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.parameters = list(parameters)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(value) for value, _ in self.parameters]
        self.v = [np.zeros_like(value) for value, _ in self.parameters]

    def step(self):
        self.t += 1
        for idx, (value, grad) in enumerate(self.parameters):
            self.m[idx] = self.beta1 * self.m[idx] + (1 - self.beta1) * grad
            self.v[idx] = self.beta2 * self.v[idx] + (1 - self.beta2) * grad**2
            m_hat = self.m[idx] / (1 - self.beta1**self.t)
            v_hat = self.v[idx] / (1 - self.beta2**self.t)
            value -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class AdamW(Adam):
    def __init__(
        self,
        parameters,
        lr=0.001,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.01,
    ):
        super().__init__(parameters, lr, beta1, beta2, eps)
        self.weight_decay = weight_decay

    def step(self):
        self.t += 1
        for idx, (value, grad) in enumerate(self.parameters):
            self.m[idx] = self.beta1 * self.m[idx] + (1 - self.beta1) * grad
            self.v[idx] = self.beta2 * self.v[idx] + (1 - self.beta2) * grad**2
            m_hat = self.m[idx] / (1 - self.beta1**self.t)
            v_hat = self.v[idx] / (1 - self.beta2**self.t)
            value *= 1 - self.lr * self.weight_decay
            value -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
