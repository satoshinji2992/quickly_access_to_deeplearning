"""A small, fully NumPy implementation of 2-D convolution.

The public layout used throughout Block 2 is NCHW.  ``im2col`` flattens one
receptive field per *row*, so its result has shape
``(N * out_h * out_w, C * kernel_h * kernel_w)``.
"""

import numpy as np


def _pair(value, name):
    if isinstance(value, (int, np.integer)):
        value = (int(value), int(value))
    else:
        try:
            value = tuple(int(item) for item in value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be an int or a pair of ints") from exc
        if len(value) != 2:
            raise ValueError(f"{name} must contain exactly two values")
    if name == "padding":
        if any(item < 0 for item in value):
            raise ValueError("padding must be non-negative")
    elif any(item <= 0 for item in value):
        raise ValueError(f"{name} must be positive")
    return value


def compute_output_size(size, kernel_size, stride, padding):
    """Return the output length for one spatial dimension."""
    size = int(size)
    kernel_size = int(kernel_size)
    stride = int(stride)
    padding = int(padding)
    if size <= 0 or kernel_size <= 0 or stride <= 0 or padding < 0:
        raise ValueError("size/kernel/stride must be positive and padding non-negative")
    output_size = (size + 2 * padding - kernel_size) // stride + 1
    if output_size <= 0:
        raise ValueError("kernel_size, stride, and padding produce an empty output")
    return output_size


def _output_hw(x_shape, kernel_size, stride, padding):
    if len(x_shape) != 4:
        raise ValueError("expected an NCHW shape with four dimensions")
    kernel_h, kernel_w = _pair(kernel_size, "kernel_size")
    stride_h, stride_w = _pair(stride, "stride")
    pad_h, pad_w = _pair(padding, "padding")
    out_h = compute_output_size(x_shape[2], kernel_h, stride_h, pad_h)
    out_w = compute_output_size(x_shape[3], kernel_w, stride_w, pad_w)
    return (kernel_h, kernel_w), (stride_h, stride_w), (pad_h, pad_w), (out_h, out_w)


def im2col(x, kernel_size, stride=1, padding=0):
    """Convert NCHW images into a 2-D matrix of receptive fields.

    Returns
    -------
    numpy.ndarray
        Shape ``(N * out_h * out_w, C * kernel_h * kernel_w)``.  Rows are
        ordered by sample, output row, then output column.  Values within a
        row are ordered by channel, kernel row, then kernel column.
    """
    x = np.asarray(x)
    if x.ndim != 4:
        raise ValueError("x must have shape (N, C, H, W)")
    (kernel_h, kernel_w), (stride_h, stride_w), (pad_h, pad_w), (out_h, out_w) = (
        _output_hw(x.shape, kernel_size, stride, padding)
    )
    x_padded = np.pad(
        x,
        ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
        mode="constant",
    )
    windows = np.lib.stride_tricks.sliding_window_view(
        x_padded, (kernel_h, kernel_w), axis=(2, 3)
    )
    windows = windows[:, :, ::stride_h, ::stride_w, :, :]
    windows = windows[:, :, :out_h, :out_w, :, :]
    n, channels = x.shape[:2]
    return windows.transpose(0, 2, 3, 1, 4, 5).reshape(
        n * out_h * out_w, channels * kernel_h * kernel_w
    )


def col2im(cols, x_shape, kernel_size, stride=1, padding=0):
    """Scatter rows produced by :func:`im2col` back into an NCHW tensor.

    Overlapping window contributions are **summed**.  Consequently,
    ``col2im(im2col(x), ...)`` equals ``x`` multiplied by the number of
    windows covering each pixel; this is exactly the accumulation needed by
    convolution backpropagation.
    """
    x_shape = tuple(int(item) for item in x_shape)
    (kernel_h, kernel_w), (stride_h, stride_w), (pad_h, pad_w), (out_h, out_w) = (
        _output_hw(x_shape, kernel_size, stride, padding)
    )
    n, channels, height, width = x_shape
    cols = np.asarray(cols)
    expected_shape = (n * out_h * out_w, channels * kernel_h * kernel_w)
    if cols.shape != expected_shape:
        raise ValueError(f"cols has shape {cols.shape}; expected {expected_shape}")

    padded = np.zeros(
        (n, channels, height + 2 * pad_h, width + 2 * pad_w), dtype=cols.dtype
    )
    windows = cols.reshape(
        n, out_h, out_w, channels, kernel_h, kernel_w
    ).transpose(0, 3, 1, 2, 4, 5)
    for kernel_row in range(kernel_h):
        row_slice = slice(kernel_row, kernel_row + stride_h * out_h, stride_h)
        for kernel_col in range(kernel_w):
            col_slice = slice(kernel_col, kernel_col + stride_w * out_w, stride_w)
            padded[:, :, row_slice, col_slice] += windows[
                :, :, :, :, kernel_row, kernel_col
            ]

    if pad_h == 0:
        row_slice = slice(None)
    else:
        row_slice = slice(pad_h, -pad_h)
    if pad_w == 0:
        col_slice = slice(None)
    else:
        col_slice = slice(pad_w, -pad_w)
    return padded[:, :, row_slice, col_slice]


class Conv2D:
    """A trainable NCHW cross-correlation layer.

    As in mainstream deep-learning libraries, the operation is technically
    cross-correlation (the stored kernel is not spatially flipped).
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        if int(in_channels) <= 0 or int(out_channels) <= 0:
            raise ValueError("in_channels and out_channels must be positive")
        kernel_h, kernel_w = _pair(kernel_size, "kernel_size")
        self.stride = _pair(stride, "stride")
        self.padding = _pair(padding, "padding")
        self.kernel_size = (kernel_h, kernel_w)
        fan_in = int(in_channels) * kernel_h * kernel_w
        scale = np.sqrt(2.0 / fan_in)
        self.W = np.random.randn(
            int(out_channels), int(in_channels), kernel_h, kernel_w
        ) * scale
        self.b = np.zeros(int(out_channels), dtype=self.W.dtype)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.x = None
        self.cols = None
        self.output_shape = None
        self.training = True

    def forward(self, x):
        x = np.asarray(x)
        if x.ndim != 4:
            raise ValueError("x must have shape (N, C, H, W)")
        if x.shape[1] != self.W.shape[1]:
            raise ValueError(
                f"x has {x.shape[1]} channels, but this layer expects {self.W.shape[1]}"
            )
        self.x = x
        self.cols = im2col(x, self.kernel_size, self.stride, self.padding)
        weight_columns = self.W.reshape(self.W.shape[0], -1).T
        flat_output = np.dot(self.cols, weight_columns) + self.b
        _, _, _, (out_h, out_w) = _output_hw(
            x.shape, self.kernel_size, self.stride, self.padding
        )
        self.output_shape = (x.shape[0], self.W.shape[0], out_h, out_w)
        return flat_output.reshape(
            x.shape[0], out_h, out_w, self.W.shape[0]
        ).transpose(0, 3, 1, 2)

    def backward(self, dout):
        if self.x is None or self.cols is None or self.output_shape is None:
            raise RuntimeError("forward must be called before backward")
        dout = np.asarray(dout)
        if dout.shape != self.output_shape:
            raise ValueError(f"dout has shape {dout.shape}; expected {self.output_shape}")
        n, out_channels, out_h, out_w = dout.shape
        flat_dout = dout.transpose(0, 2, 3, 1).reshape(
            n * out_h * out_w, out_channels
        )

        # Keep gradient array identities stable: optimizers store references to
        # these arrays when they are constructed.
        self.db[...] = np.sum(flat_dout, axis=0)
        self.dW[...] = np.dot(self.cols.T, flat_dout).T.reshape(self.W.shape)
        dx_columns = np.dot(flat_dout, self.W.reshape(out_channels, -1))
        return col2im(
            dx_columns,
            self.x.shape,
            self.kernel_size,
            self.stride,
            self.padding,
        )

    def parameters(self):
        return [(self.W, self.dW), (self.b, self.db)]

    def named_parameters(self, prefix=""):
        prefix = f"{prefix}." if prefix else ""
        return [
            (f"{prefix}W", self.W, self.dW),
            (f"{prefix}b", self.b, self.db),
        ]

    def named_buffers(self, prefix=""):
        return []

    def train(self):
        self.training = True
        return self

    def eval(self):
        self.training = False
        return self
