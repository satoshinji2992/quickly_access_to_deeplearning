"""Reference exports for Block 2.

The exercise files now contain complete, tested implementations.  Re-exporting
them here keeps the reference answer and the code students run from silently
diverging again.
"""

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from common.my_dl_lib import CrossEntropyLoss, Linear, Momentum, ReLU  # noqa: E402,F401
from exercises.block_02_resnet.task_10_image_data_pipeline.data_pipeline import (  # noqa: E402,F401
    CIFAR100_MEAN,
    CIFAR100_STD,
    assert_disjoint_splits,
    iterate_minibatches,
    load_cifar100_splits,
    normalize,
    random_crop_with_padding,
    random_horizontal_flip,
    stratified_subset,
    stratified_train_val_split,
    to_nchw,
    validate_splits,
)
from exercises.block_02_resnet.task_11_conv2d_im2col.conv2d import (  # noqa: E402,F401
    Conv2D,
    col2im,
    compute_output_size,
    im2col,
)
from exercises.block_02_resnet.task_12_pooling_and_bn.layers import (  # noqa: E402,F401
    BatchNorm2D,
    GlobalAvgPool2D,
    MaxPool2D,
)
from exercises.block_02_resnet.task_13_residual_block.residual_block import (  # noqa: E402,F401
    BasicBlock,
)
from exercises.block_02_resnet.task_14_numpy_resnet_train.train_resnet import (  # noqa: E402,F401
    SmallResNet,
    accuracy,
    augment_batch,
    evaluate,
    make_synthetic_splits,
    one_hot,
    train_epoch,
)


__all__ = [
    "BasicBlock",
    "BatchNorm2D",
    "CIFAR100_MEAN",
    "CIFAR100_STD",
    "Conv2D",
    "CrossEntropyLoss",
    "GlobalAvgPool2D",
    "Linear",
    "MaxPool2D",
    "Momentum",
    "ReLU",
    "SmallResNet",
    "accuracy",
    "assert_disjoint_splits",
    "augment_batch",
    "col2im",
    "compute_output_size",
    "evaluate",
    "im2col",
    "iterate_minibatches",
    "load_cifar100_splits",
    "make_synthetic_splits",
    "normalize",
    "one_hot",
    "random_crop_with_padding",
    "random_horizontal_flip",
    "stratified_subset",
    "stratified_train_val_split",
    "to_nchw",
    "train_epoch",
    "validate_splits",
]
