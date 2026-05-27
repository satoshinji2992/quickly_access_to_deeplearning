import numpy as np


def compute_output_size(size, kernel_size, stride, padding):
    return (size + 2 * padding - kernel_size) // stride + 1


def im2col(x, kernel_size, stride=1, padding=0):
    """Convert NCHW images into columns.

    TODO: implement this in the exercise.
    Expected output shape:
    (N * out_h * out_w, C * kernel_size * kernel_size)
    """
    raise NotImplementedError("implement im2col")


def col2im(cols, x_shape, kernel_size, stride=1, padding=0):
    """Reverse im2col.

    TODO: implement this in the exercise.
    """
    raise NotImplementedError("implement col2im")


class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.W = (
            np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale
        )
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
        self.db = np.sum(dout_col, axis=0)
        self.dW = (self.cols.T @ dout_col).T.reshape(self.W.shape)
        w_col = self.W.reshape(out_c, -1)
        dx_cols = dout_col @ w_col
        return col2im(dx_cols, self.x.shape, self.kernel_size, self.stride, self.padding)

