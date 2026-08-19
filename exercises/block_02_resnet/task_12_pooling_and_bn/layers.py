"""Pooling and normalization layers used by the NumPy ResNet exercises."""

import numpy as np


def _pair(value, name):
    if isinstance(value, (int, np.integer)):
        result = (int(value), int(value))
    else:
        try:
            result = tuple(int(item) for item in value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be an int or a pair of ints") from exc
        if len(result) != 2:
            raise ValueError(f"{name} must contain exactly two values")
    if any(item <= 0 for item in result):
        raise ValueError(f"{name} must be positive")
    return result


def _output_size(size, kernel, stride):
    output = (int(size) - int(kernel)) // int(stride) + 1
    if output <= 0:
        raise ValueError("pooling window produces an empty output")
    return output


class MaxPool2D:
    """NCHW max pooling.

    Ties follow NumPy's ``argmax`` convention: the first maximum in row-major
    order receives the complete upstream gradient.  This deterministic policy
    matches the common single-index max-pool implementation.
    """

    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = _pair(kernel_size, "kernel_size")
        self.stride = _pair(stride, "stride")
        self.x_shape = None
        self.argmax = None
        self.output_shape = None
        self.training = True

    def forward(self, x):
        x = np.asarray(x)
        if x.ndim != 4:
            raise ValueError("x must have shape (N, C, H, W)")
        n, channels, height, width = x.shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        out_h = _output_size(height, kernel_h, stride_h)
        out_w = _output_size(width, kernel_w, stride_w)
        output = np.empty((n, channels, out_h, out_w), dtype=x.dtype)
        self.argmax = np.empty((n, channels, out_h, out_w), dtype=np.int64)
        self.x_shape = x.shape
        self.output_shape = output.shape

        for output_row in range(out_h):
            top = output_row * stride_h
            for output_col in range(out_w):
                left = output_col * stride_w
                window = x[
                    :,
                    :,
                    top : top + kernel_h,
                    left : left + kernel_w,
                ].reshape(n, channels, -1)
                winner = np.argmax(window, axis=2)
                self.argmax[:, :, output_row, output_col] = winner
                output[:, :, output_row, output_col] = np.take_along_axis(
                    window, winner[:, :, None], axis=2
                )[:, :, 0]
        return output

    def backward(self, dout):
        if self.x_shape is None or self.argmax is None:
            raise RuntimeError("forward must be called before backward")
        dout = np.asarray(dout)
        if dout.shape != self.output_shape:
            raise ValueError(f"dout has shape {dout.shape}; expected {self.output_shape}")
        n, channels, height, width = self.x_shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        out_h, out_w = dout.shape[2:]
        dx = np.zeros((n, channels, height, width), dtype=dout.dtype)
        sample_index = np.arange(n)[:, None]
        channel_index = np.arange(channels)[None, :]
        for output_row in range(out_h):
            top = output_row * stride_h
            for output_col in range(out_w):
                left = output_col * stride_w
                winner = self.argmax[:, :, output_row, output_col]
                winner_row = winner // kernel_w
                winner_col = winner % kernel_w
                np.add.at(
                    dx,
                    (
                        sample_index,
                        channel_index,
                        top + winner_row,
                        left + winner_col,
                    ),
                    dout[:, :, output_row, output_col],
                )
        return dx

    def parameters(self):
        return []

    def named_parameters(self, prefix=""):
        return []

    def named_buffers(self, prefix=""):
        return []

    def train(self):
        self.training = True
        return self

    def eval(self):
        self.training = False
        return self


class GlobalAvgPool2D:
    def __init__(self):
        self.x_shape = None
        self.training = True

    def forward(self, x):
        x = np.asarray(x)
        if x.ndim != 4:
            raise ValueError("x must have shape (N, C, H, W)")
        self.x_shape = x.shape
        return np.mean(x, axis=(2, 3))

    def backward(self, dout):
        if self.x_shape is None:
            raise RuntimeError("forward must be called before backward")
        dout = np.asarray(dout)
        n, channels, height, width = self.x_shape
        expected_shape = (n, channels)
        if dout.shape != expected_shape:
            raise ValueError(f"dout has shape {dout.shape}; expected {expected_shape}")
        return np.broadcast_to(
            dout[:, :, None, None] / (height * width), self.x_shape
        ).copy()

    def parameters(self):
        return []

    def named_parameters(self, prefix=""):
        return []

    def named_buffers(self, prefix=""):
        return []

    def train(self):
        self.training = True
        return self

    def eval(self):
        self.training = False
        return self


class BatchNorm2D:
    """Batch normalization over the channel axis of an NCHW tensor."""

    def __init__(self, channels, momentum=0.9, eps=1e-5):
        channels = int(channels)
        if channels <= 0:
            raise ValueError("channels must be positive")
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("momentum must be in [0, 1]")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.gamma = np.ones((1, channels, 1, 1))
        self.beta = np.zeros((1, channels, 1, 1))
        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta = np.zeros_like(self.beta)
        self.running_mean = np.zeros((1, channels, 1, 1))
        self.running_var = np.ones((1, channels, 1, 1))
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.training = True
        self.x_hat = None
        self.std_inv = None
        self._forward_was_training = None
        self._input_shape = None

    def train(self):
        self.training = True
        return self

    def eval(self):
        self.training = False
        return self

    def forward(self, x):
        x = np.asarray(x)
        if x.ndim != 4:
            raise ValueError("x must have shape (N, C, H, W)")
        if x.shape[1] != self.gamma.shape[1]:
            raise ValueError(
                f"x has {x.shape[1]} channels, but this layer expects {self.gamma.shape[1]}"
            )
        axes = (0, 2, 3)
        if self.training:
            mean = np.mean(x, axis=axes, keepdims=True)
            variance = np.var(x, axis=axes, keepdims=True)
            # Preserve buffer identities so state-dict/checkpoint code can keep
            # stable references to them.
            self.running_mean[...] = (
                self.momentum * self.running_mean + (1.0 - self.momentum) * mean
            )
            self.running_var[...] = (
                self.momentum * self.running_var + (1.0 - self.momentum) * variance
            )
        else:
            mean = self.running_mean
            variance = self.running_var
        self.std_inv = 1.0 / np.sqrt(variance + self.eps)
        self.x_hat = (x - mean) * self.std_inv
        self._forward_was_training = self.training
        self._input_shape = x.shape
        return self.gamma * self.x_hat + self.beta

    def backward(self, dout):
        if self.x_hat is None or self.std_inv is None or self._input_shape is None:
            raise RuntimeError("forward must be called before backward")
        dout = np.asarray(dout)
        if dout.shape != self._input_shape:
            raise ValueError(f"dout has shape {dout.shape}; expected {self._input_shape}")
        axes = (0, 2, 3)
        self.dbeta[...] = np.sum(dout, axis=axes, keepdims=True)
        self.dgamma[...] = np.sum(dout * self.x_hat, axis=axes, keepdims=True)
        dx_hat = dout * self.gamma
        if not self._forward_was_training:
            # Evaluation statistics are constants, so no mean/variance terms
            # participate in this derivative.
            return dx_hat * self.std_inv
        element_count = dout.shape[0] * dout.shape[2] * dout.shape[3]
        sum_dx_hat = np.sum(dx_hat, axis=axes, keepdims=True)
        sum_dx_hat_x_hat = np.sum(dx_hat * self.x_hat, axis=axes, keepdims=True)
        return (
            dx_hat
            - sum_dx_hat / element_count
            - self.x_hat * sum_dx_hat_x_hat / element_count
        ) * self.std_inv

    def parameters(self):
        return [(self.gamma, self.dgamma), (self.beta, self.dbeta)]

    def named_parameters(self, prefix=""):
        prefix = f"{prefix}." if prefix else ""
        return [
            (f"{prefix}gamma", self.gamma, self.dgamma),
            (f"{prefix}beta", self.beta, self.dbeta),
        ]

    def named_buffers(self, prefix=""):
        prefix = f"{prefix}." if prefix else ""
        return [
            (f"{prefix}running_mean", self.running_mean),
            (f"{prefix}running_var", self.running_var),
        ]
