class BasicBlock:
    """NumPy ResNet BasicBlock skeleton.

    Expected structure:
    Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> shortcut add -> ReLU
    """

    def __init__(self, in_channels, out_channels, stride=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.needs_projection = stride != 1 or in_channels != out_channels
        # TODO: create conv/bn/relu layers after task_11 and task_12 are done.

    def forward(self, x):
        raise NotImplementedError("implement BasicBlock forward")

    def backward(self, dout):
        raise NotImplementedError("implement BasicBlock backward")

