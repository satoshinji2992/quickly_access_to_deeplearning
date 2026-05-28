import numpy as np


def compute_output_size(size, kernel_size, stride, padding):
    return (size + 2 * padding - kernel_size) // stride + 1


def to_nchw(images):
    images = images.astype("float32")
    if images.ndim != 4:
        raise ValueError("expected images with shape (N, H, W, C)")
    return images.transpose(0, 3, 1, 2)


def normalize(images, mean, std):
    mean = np.asarray(mean, dtype="float32").reshape(1, 3, 1, 1)
    std = np.asarray(std, dtype="float32").reshape(1, 3, 1, 1)
    return (images / 255.0 - mean) / std


def iterate_minibatches(images, labels, batch_size, shuffle=True, seed=None):
    rng = np.random.default_rng(seed)
    indices = np.arange(images.shape[0])
    if shuffle:
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        idx = indices[start : start + batch_size]
        yield images[idx], labels[idx]


def im2col(x, kernel_size, stride=1, padding=0):
    n, c, h, w = x.shape
    out_h = compute_output_size(h, kernel_size, stride, padding)
    out_w = compute_output_size(w, kernel_size, stride, padding)
    x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    cols = np.empty((n * out_h * out_w, c * kernel_size * kernel_size), dtype=x.dtype)

    row = 0
    for sample in range(n):
        for i in range(out_h):
            top = i * stride
            for j in range(out_w):
                left = j * stride
                window = x_padded[
                    sample,
                    :,
                    top : top + kernel_size,
                    left : left + kernel_size,
                ]
                cols[row] = window.reshape(-1)
                row += 1
    return cols


def col2im(cols, x_shape, kernel_size, stride=1, padding=0):
    n, c, h, w = x_shape
    out_h = compute_output_size(h, kernel_size, stride, padding)
    out_w = compute_output_size(w, kernel_size, stride, padding)
    h_padded = h + 2 * padding
    w_padded = w + 2 * padding
    x_padded = np.zeros((n, c, h_padded, w_padded), dtype=cols.dtype)

    row = 0
    for sample in range(n):
        for i in range(out_h):
            top = i * stride
            for j in range(out_w):
                left = j * stride
                window = cols[row].reshape(c, kernel_size, kernel_size)
                x_padded[
                    sample,
                    :,
                    top : top + kernel_size,
                    left : left + kernel_size,
                ] += window
                row += 1

    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]


class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = x > 0
        return np.maximum(x, 0)

    def backward(self, dout):
        return dout * self.mask

    def parameters(self):
        return []

    def train(self):
        pass

    def eval(self):
        pass


class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale
        self.b = np.zeros(out_channels)
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.x = None
        self.cols = None
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        self.cols = im2col(x, self.kernel_size, self.stride, self.padding)
        w_col = self.W.reshape(self.W.shape[0], -1).T
        out = self.cols @ w_col + self.b
        n, _, h, w = x.shape
        out_h = compute_output_size(h, self.kernel_size, self.stride, self.padding)
        out_w = compute_output_size(w, self.kernel_size, self.stride, self.padding)
        return out.reshape(n, out_h, out_w, self.W.shape[0]).transpose(0, 3, 1, 2)

    def backward(self, dout):
        n, out_c, out_h, out_w = dout.shape
        dout_col = dout.transpose(0, 2, 3, 1).reshape(n * out_h * out_w, out_c)
        self.db[...] = np.sum(dout_col, axis=0)
        self.dW[...] = (self.cols.T @ dout_col).T.reshape(self.W.shape)
        w_col = self.W.reshape(out_c, -1)
        dx_cols = dout_col @ w_col
        return col2im(dx_cols, self.x.shape, self.kernel_size, self.stride, self.padding)

    def parameters(self):
        return [(self.W, self.dW), (self.b, self.db)]

    def train(self):
        pass

    def eval(self):
        pass


class MaxPool2D:
    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride
        self.x_shape = None
        self.argmax = None

    def forward(self, x):
        n, c, h, w = x.shape
        out_h = compute_output_size(h, self.kernel_size, self.stride, 0)
        out_w = compute_output_size(w, self.kernel_size, self.stride, 0)
        out = np.empty((n, c, out_h, out_w), dtype=x.dtype)
        self.argmax = np.empty((n, c, out_h, out_w), dtype=np.int64)
        self.x_shape = x.shape

        for i in range(out_h):
            top = i * self.stride
            for j in range(out_w):
                left = j * self.stride
                window = x[:, :, top : top + self.kernel_size, left : left + self.kernel_size]
                flat = window.reshape(n, c, -1)
                self.argmax[:, :, i, j] = np.argmax(flat, axis=2)
                out[:, :, i, j] = np.max(flat, axis=2)
        return out

    def backward(self, dout):
        n, c, h, w = self.x_shape
        _, _, out_h, out_w = dout.shape
        dx = np.zeros((n, c, h, w), dtype=dout.dtype)
        for i in range(out_h):
            top = i * self.stride
            for j in range(out_w):
                left = j * self.stride
                arg = self.argmax[:, :, i, j]
                row = arg // self.kernel_size
                col = arg % self.kernel_size
                for sample in range(n):
                    for ch in range(c):
                        dx[sample, ch, top + row[sample, ch], left + col[sample, ch]] += dout[
                            sample, ch, i, j
                        ]
        return dx

    def parameters(self):
        return []

    def train(self):
        pass

    def eval(self):
        pass


class GlobalAvgPool2D:
    def forward(self, x):
        self.x_shape = x.shape
        return np.mean(x, axis=(2, 3))

    def backward(self, dout):
        n, c, h, w = self.x_shape
        return dout[:, :, None, None] * np.ones((n, c, h, w), dtype=dout.dtype) / (h * w)

    def parameters(self):
        return []

    def train(self):
        pass

    def eval(self):
        pass


class BatchNorm2D:
    def __init__(self, channels, momentum=0.9, eps=1e-5):
        self.gamma = np.ones((1, channels, 1, 1))
        self.beta = np.zeros((1, channels, 1, 1))
        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta = np.zeros_like(self.beta)
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

        self.x = x
        self.mean = mean
        self.var = var
        self.x_centered = x - mean
        self.std_inv = 1.0 / np.sqrt(var + self.eps)
        self.x_hat = self.x_centered * self.std_inv
        return self.gamma * self.x_hat + self.beta

    def backward(self, dout):
        axes = (0, 2, 3)
        m = dout.shape[0] * dout.shape[2] * dout.shape[3]
        self.dbeta[...] = np.sum(dout, axis=axes, keepdims=True)
        self.dgamma[...] = np.sum(dout * self.x_hat, axis=axes, keepdims=True)

        dxhat = dout * self.gamma
        sum_dxhat = np.sum(dxhat, axis=axes, keepdims=True)
        sum_dxhat_xhat = np.sum(dxhat * self.x_hat, axis=axes, keepdims=True)
        dx = (dxhat - sum_dxhat / m - self.x_hat * sum_dxhat_xhat / m) * self.std_inv
        return dx

    def parameters(self):
        return [(self.gamma, self.dgamma), (self.beta, self.dbeta)]


class Linear:
    def __init__(self, in_features, out_features):
        scale = np.sqrt(2.0 / in_features)
        self.W = np.random.randn(in_features, out_features) * scale
        self.b = np.zeros((1, out_features))
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout):
        self.dW[...] = self.x.T @ dout
        self.db[...] = np.sum(dout, axis=0, keepdims=True)
        return dout @ self.W.T

    def parameters(self):
        return [(self.W, self.dW), (self.b, self.db)]

    def train(self):
        pass

    def eval(self):
        pass


class CrossEntropyLoss:
    def forward(self, logits, labels):
        self.labels = labels
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(shifted)
        self.probs = exp / np.sum(exp, axis=1, keepdims=True)
        return -np.mean(np.log(self.probs[np.arange(labels.shape[0]), labels] + 1e-12))

    def backward(self):
        grad = self.probs.copy()
        grad[np.arange(self.labels.shape[0]), self.labels] -= 1.0
        return grad / self.labels.shape[0]


class MomentumSGD:
    def __init__(self, parameters, lr=0.01, beta=0.9, weight_decay=0.0):
        self.parameters = list(parameters)
        self.lr = lr
        self.beta = beta
        self.weight_decay = weight_decay
        self.velocity = [np.zeros_like(value) for value, _ in self.parameters]

    def step(self):
        for i, (value, grad) in enumerate(self.parameters):
            update = grad + self.weight_decay * value
            self.velocity[i] = self.beta * self.velocity[i] + update
            value -= self.lr * self.velocity[i]


class BasicBlock:
    def __init__(self, in_channels, out_channels, stride=1):
        self.conv1 = Conv2D(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = BatchNorm2D(out_channels)
        self.relu1 = ReLU()
        self.conv2 = Conv2D(out_channels, out_channels, 3, stride=1, padding=1)
        self.bn2 = BatchNorm2D(out_channels)
        self.relu2 = ReLU()
        self.needs_projection = stride != 1 or in_channels != out_channels
        if self.needs_projection:
            self.proj_conv = Conv2D(in_channels, out_channels, 1, stride=stride, padding=0)
            self.proj_bn = BatchNorm2D(out_channels)

    def forward(self, x):
        self.x = x
        out = self.conv1.forward(x)
        out = self.bn1.forward(out)
        out = self.relu1.forward(out)
        out = self.conv2.forward(out)
        out = self.bn2.forward(out)
        if self.needs_projection:
            shortcut = self.proj_bn.forward(self.proj_conv.forward(x))
        else:
            shortcut = x
        self.added = out + shortcut
        return self.relu2.forward(self.added)

    def backward(self, dout):
        dz = self.relu2.backward(dout)
        dmain = self.bn2.backward(dz)
        dmain = self.conv2.backward(dmain)
        dmain = self.relu1.backward(dmain)
        dmain = self.bn1.backward(dmain)
        dmain = self.conv1.backward(dmain)
        if self.needs_projection:
            dshortcut = self.proj_bn.backward(dz)
            dshortcut = self.proj_conv.backward(dshortcut)
        else:
            dshortcut = dz
        return dmain + dshortcut

    def parameters(self):
        params = []
        for layer in [self.conv1, self.bn1, self.conv2, self.bn2]:
            params.extend(layer.parameters())
        if self.needs_projection:
            params.extend(self.proj_conv.parameters())
            params.extend(self.proj_bn.parameters())
        return params

    def train(self):
        for layer in self._layers():
            layer.train()

    def eval(self):
        for layer in self._layers():
            layer.eval()

    def _layers(self):
        layers = [self.conv1, self.bn1, self.relu1, self.conv2, self.bn2, self.relu2]
        if self.needs_projection:
            layers += [self.proj_conv, self.proj_bn]
        return layers


class SmallResNet:
    def __init__(self, num_classes=100, channels=(16, 32, 64)):
        c1, c2, c3 = channels
        self.stem = [Conv2D(3, c1, 3, padding=1), BatchNorm2D(c1), ReLU()]
        self.blocks = [
            BasicBlock(c1, c1),
            BasicBlock(c1, c2, stride=2),
            BasicBlock(c2, c3, stride=2),
        ]
        self.pool = GlobalAvgPool2D()
        self.fc = Linear(c3, num_classes)

    def forward(self, x):
        for layer in self.stem:
            x = layer.forward(x)
        for block in self.blocks:
            x = block.forward(x)
        x = self.pool.forward(x)
        return self.fc.forward(x)

    def backward(self, dout):
        dout = self.fc.backward(dout)
        dout = self.pool.backward(dout)
        for block in reversed(self.blocks):
            dout = block.backward(dout)
        for layer in reversed(self.stem):
            dout = layer.backward(dout)
        return dout

    def parameters(self):
        params = []
        for layer in self.stem:
            params.extend(layer.parameters())
        for block in self.blocks:
            params.extend(block.parameters())
        params.extend(self.pool.parameters())
        params.extend(self.fc.parameters())
        return params

    def train(self):
        for layer in self.stem:
            layer.train()
        for block in self.blocks:
            block.train()

    def eval(self):
        for layer in self.stem:
            layer.eval()
        for block in self.blocks:
            block.eval()


def accuracy(logits, labels):
    return np.mean(np.argmax(logits, axis=1) == labels)


def make_toy_images(n=24, num_classes=3, seed=0):
    rng = np.random.default_rng(seed)
    images = rng.normal(0, 0.15, size=(n, 3, 16, 16)).astype("float32")
    labels = np.arange(n) % num_classes
    for i, label in enumerate(labels):
        images[i, label % 3, 3 + label : 8 + label, 3 + label : 8 + label] += 1.0
    return images, labels.astype("int64")


def smoke_train():
    np.random.seed(0)
    x, y = make_toy_images()
    model = SmallResNet(num_classes=3, channels=(4, 8, 8))
    loss_fn = CrossEntropyLoss()
    optimizer = MomentumSGD(model.parameters(), lr=0.03, beta=0.9)
    for step in range(8):
        model.train()
        logits = model.forward(x)
        loss = loss_fn.forward(logits, y)
        model.backward(loss_fn.backward())
        optimizer.step()
        print(f"step={step} loss={loss:.4f} acc={accuracy(logits, y):.3f}")


if __name__ == "__main__":
    smoke_train()
