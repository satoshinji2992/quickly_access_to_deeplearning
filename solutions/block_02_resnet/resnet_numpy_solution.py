import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from common.my_dl_lib import CrossEntropyLoss, Linear, Momentum, ReLU  # noqa: E402


def _parameters(layer):
    if hasattr(layer, "parameters"):
        return layer.parameters()
    return []


def _train(layer):
    if hasattr(layer, "train"):
        layer.train()


def _eval(layer):
    if hasattr(layer, "eval"):
        layer.eval()


def compute_output_size(size, kernel_size, stride, padding):
    out = (size + 2 * padding - kernel_size) // stride + 1
    if out <= 0:
        raise ValueError("kernel_size, stride, and padding produce an empty output")
    return out


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
    windows = np.lib.stride_tricks.sliding_window_view(
        x_padded, (kernel_size, kernel_size), axis=(2, 3)
    )
    windows = windows[:, :, ::stride, ::stride, :, :]
    windows = windows[:, :, :out_h, :out_w, :, :]
    return windows.transpose(0, 2, 3, 1, 4, 5).reshape(
        n * out_h * out_w, c * kernel_size * kernel_size
    )


def col2im(cols, x_shape, kernel_size, stride=1, padding=0):
    n, c, h, w = x_shape
    out_h = compute_output_size(h, kernel_size, stride, padding)
    out_w = compute_output_size(w, kernel_size, stride, padding)
    h_padded = h + 2 * padding
    w_padded = w + 2 * padding
    x_padded = np.zeros((n, c, h_padded, w_padded), dtype=cols.dtype)
    windows = cols.reshape(n, out_h, out_w, c, kernel_size, kernel_size).transpose(
        0, 3, 1, 2, 4, 5
    )

    for kh in range(kernel_size):
        h_slice = slice(kh, kh + stride * out_h, stride)
        for kw in range(kernel_size):
            w_slice = slice(kw, kw + stride * out_w, stride)
            x_padded[:, :, h_slice, w_slice] += windows[:, :, :, :, kh, kw]

    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]


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
        out = np.dot(self.cols, w_col) + self.b
        n, _, h, w = x.shape
        out_h = compute_output_size(h, self.kernel_size, self.stride, self.padding)
        out_w = compute_output_size(w, self.kernel_size, self.stride, self.padding)
        return out.reshape(n, out_h, out_w, self.W.shape[0]).transpose(0, 3, 1, 2)

    def backward(self, dout):
        n, out_c, out_h, out_w = dout.shape
        dout_col = dout.transpose(0, 2, 3, 1).reshape(n * out_h * out_w, out_c)
        self.db[...] = np.sum(dout_col, axis=0)
        self.dW[...] = np.dot(self.cols.T, dout_col).T.reshape(self.W.shape)
        w_col = self.W.reshape(out_c, -1)
        dx_cols = np.dot(dout_col, w_col)
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
        sample_idx = np.arange(n)[:, None]
        channel_idx = np.arange(c)[None, :]
        for i in range(out_h):
            top = i * self.stride
            for j in range(out_w):
                left = j * self.stride
                arg = self.argmax[:, :, i, j]
                row = arg // self.kernel_size
                col = arg % self.kernel_size
                np.add.at(
                    dx,
                    (sample_idx, channel_idx, top + row, left + col),
                    dout[:, :, i, j],
                )
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
            _train(layer)

    def eval(self):
        for layer in self._layers():
            _eval(layer)

    def _layers(self):
        layers = [self.conv1, self.bn1, self.relu1, self.conv2, self.bn2, self.relu2]
        if self.needs_projection:
            layers += [self.proj_conv, self.proj_bn]
        return layers


class SmallResNet:
    def __init__(self, num_classes=100, channels=(16, 32, 64), blocks_per_stage=(2, 2, 2)):
        c1, c2, c3 = channels
        self.stem = [Conv2D(3, c1, 3, padding=1), BatchNorm2D(c1), ReLU()]
        self.blocks = []
        in_channels = c1
        for stage_idx, (out_channels, block_count) in enumerate(
            zip(channels, blocks_per_stage)
        ):
            for block_idx in range(block_count):
                stride = 2 if stage_idx > 0 and block_idx == 0 else 1
                self.blocks.append(BasicBlock(in_channels, out_channels, stride=stride))
                in_channels = out_channels
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
            params.extend(_parameters(layer))
        for block in self.blocks:
            params.extend(block.parameters())
        params.extend(_parameters(self.pool))
        params.extend(self.fc.parameters())
        return params

    def train(self):
        for layer in self.stem:
            _train(layer)
        for block in self.blocks:
            block.train()

    def eval(self):
        for layer in self.stem:
            _eval(layer)
        for block in self.blocks:
            block.eval()


def accuracy(logits, labels):
    return np.mean(np.argmax(logits, axis=1) == labels)


def make_training_images(n=24, num_classes=3, seed=0):
    rng = np.random.default_rng(seed)
    images = rng.normal(0, 0.15, size=(n, 3, 16, 16)).astype("float32")
    labels = np.arange(n) % num_classes
    for i, label in enumerate(labels):
        images[i, label % 3, 3 + label : 8 + label, 3 + label : 8 + label] += 1.0
    return images, labels.astype("int64")


def one_hot(labels, num_classes):
    y = np.zeros((labels.shape[0], num_classes))
    y[np.arange(labels.shape[0]), labels] = 1
    return y


def train():
    np.random.seed(0)
    x, y = make_training_images()
    model = SmallResNet(num_classes=3, channels=(4, 8, 8))
    loss_fn = CrossEntropyLoss()
    optimizer = Momentum(model.parameters(), lr=0.3, beta=0.9)
    targets = one_hot(y, num_classes=3)
    for step in range(8):
        model.train()
        logits = model.forward(x)
        loss = loss_fn.forward(logits, targets)
        model.backward(loss_fn.backward())
        optimizer.step()
        print(f"step={step} loss={loss:.4f} acc={accuracy(logits, y):.3f}")


if __name__ == "__main__":
    train()
