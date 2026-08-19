"""A complete NumPy implementation of a ResNet BasicBlock."""

import sys
from pathlib import Path

try:
    from common.my_dl_lib import ReLU
    from exercises.block_02_resnet.task_11_conv2d_im2col.conv2d import Conv2D
    from exercises.block_02_resnet.task_12_pooling_and_bn.layers import BatchNorm2D
except ModuleNotFoundError:  # Allow ``python residual_block.py`` from any cwd.
    repository_root = Path(__file__).resolve().parents[3]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))
    from common.my_dl_lib import ReLU
    from exercises.block_02_resnet.task_11_conv2d_im2col.conv2d import Conv2D
    from exercises.block_02_resnet.task_12_pooling_and_bn.layers import BatchNorm2D


def _parameters(layer):
    return layer.parameters() if hasattr(layer, "parameters") else []


def _set_mode(layer, training):
    method = getattr(layer, "train" if training else "eval", None)
    if method is not None:
        method()


class BasicBlock:
    """``Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> add -> ReLU``.

    A 1x1 convolution followed by BatchNorm is used on the shortcut whenever
    the spatial stride or channel count changes.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.stride = int(stride)
        if self.in_channels <= 0 or self.out_channels <= 0 or self.stride <= 0:
            raise ValueError("channels and stride must be positive")

        self.conv1 = Conv2D(
            self.in_channels,
            self.out_channels,
            kernel_size=3,
            stride=self.stride,
            padding=1,
        )
        self.bn1 = BatchNorm2D(self.out_channels)
        self.relu1 = ReLU()
        self.conv2 = Conv2D(
            self.out_channels,
            self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bn2 = BatchNorm2D(self.out_channels)
        self.relu2 = ReLU()

        self.needs_projection = self.stride != 1 or self.in_channels != self.out_channels
        self.proj_conv = None
        self.proj_bn = None
        if self.needs_projection:
            self.proj_conv = Conv2D(
                self.in_channels,
                self.out_channels,
                kernel_size=1,
                stride=self.stride,
                padding=0,
            )
            self.proj_bn = BatchNorm2D(self.out_channels)
        self.training = True
        self._has_forward_cache = False

    def forward(self, x):
        main = self.conv1.forward(x)
        main = self.bn1.forward(main)
        main = self.relu1.forward(main)
        main = self.conv2.forward(main)
        main = self.bn2.forward(main)

        if self.needs_projection:
            shortcut = self.proj_conv.forward(x)
            shortcut = self.proj_bn.forward(shortcut)
        else:
            shortcut = x
        if main.shape != shortcut.shape:
            raise RuntimeError(
                f"main path shape {main.shape} does not match shortcut {shortcut.shape}"
            )
        self._has_forward_cache = True
        return self.relu2.forward(main + shortcut)

    def backward(self, dout):
        if not self._has_forward_cache:
            raise RuntimeError("forward must be called before backward")
        dadded = self.relu2.backward(dout)

        dmain = self.bn2.backward(dadded)
        dmain = self.conv2.backward(dmain)
        dmain = self.relu1.backward(dmain)
        dmain = self.bn1.backward(dmain)
        dmain = self.conv1.backward(dmain)

        if self.needs_projection:
            dshortcut = self.proj_bn.backward(dadded)
            dshortcut = self.proj_conv.backward(dshortcut)
        else:
            dshortcut = dadded
        return dmain + dshortcut

    def _parameter_layers(self):
        layers = [
            ("conv1", self.conv1),
            ("bn1", self.bn1),
            ("conv2", self.conv2),
            ("bn2", self.bn2),
        ]
        if self.needs_projection:
            layers.extend([("proj_conv", self.proj_conv), ("proj_bn", self.proj_bn)])
        return layers

    def _all_layers(self):
        layers = [self.conv1, self.bn1, self.relu1, self.conv2, self.bn2, self.relu2]
        if self.needs_projection:
            layers.extend([self.proj_conv, self.proj_bn])
        return layers

    def parameters(self):
        parameters = []
        for _, layer in self._parameter_layers():
            parameters.extend(_parameters(layer))
        return parameters

    def named_parameters(self, prefix=""):
        named = []
        for layer_name, layer in self._parameter_layers():
            child_prefix = ".".join(part for part in (prefix, layer_name) if part)
            named.extend(layer.named_parameters(child_prefix))
        return named

    def named_buffers(self, prefix=""):
        named = []
        for layer_name, layer in self._parameter_layers():
            if not hasattr(layer, "named_buffers"):
                continue
            child_prefix = ".".join(part for part in (prefix, layer_name) if part)
            named.extend(layer.named_buffers(child_prefix))
        return named

    def train(self):
        self.training = True
        for layer in self._all_layers():
            _set_mode(layer, training=True)
        return self

    def eval(self):
        self.training = False
        for layer in self._all_layers():
            _set_mode(layer, training=False)
        return self
