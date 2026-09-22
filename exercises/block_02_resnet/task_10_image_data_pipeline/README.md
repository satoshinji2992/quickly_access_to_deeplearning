# 图片进入卷积层之前

卷积代码约定输入 shape 为 `(N,C,H,W)`，CIFAR-100 下载下来的数组却是 `(N,H,W,C)`，数值还是 0 到 255 的整数。数据管线要把这些差别一次处理好，并保证图片和标签始终成对移动。[`data_pipeline.py`](./data_pipeline.py) 按一批图片真正经过的顺序实现了这条路径。

![图像数据管线中的处理项](assets/image_data_pipeline.png)

## 先把官方训练集分成两份

CIFAR-100 有 100 个细类别。官方数据规模是：

```text
train  50,000 张，每类 500 张
test   10,000 张，每类 100 张
```

官方没有单独提供验证集。本仓库从 50,000 张 train 中分出 5,000 张 validation，官方 test 不参与训练和配置选择：

```text
官方 train 50,000
├── train       45,000
└── validation   5,000

官方 test       10,000
```

`stratified_train_val_split()` 按类别拆分，默认让每类 450 张进入训练集、50 张进入验证集。它用两组不相交的源索引取数据，而不是先拿出一段验证数据再让训练集继续包含这些样本。

为了缩短运行时间，`train_limit`、`val_limit` 和 `test_limit` 可以在各自 split 内再次分层抽样，但不会把测试图片移进训练集。即使没有保留原始索引，`assert_disjoint_splits()` 也会比较图片内容的指纹，抓出“验证集其实是训练集切片”这类泄漏。

## 把 NHWC 换成 NCHW

`torchvision.datasets.CIFAR100.data` 把通道放在最后：

```text
(N,32,32,3)   # NHWC
```

本章的卷积层把通道放在空间轴前面：

```text
(N,3,32,32)   # NCHW
```

![NHWC 与 NCHW 只是轴顺序不同](assets/nhwc_nchw.png)

`to_nchw()` 做的事情很直接：

```python
images.astype("float32", copy=False).transpose(0, 3, 1, 2)
```

`transpose` 只改变轴的顺序，不会改变像素值。比如原来的 `images[0,5,7,2]` 是第 0 张图、第 5 行、第 7 列的蓝色通道，变换后应该从 `images_nchw[0,2,5,7]` 读到它。直接写 `reshape(N,3,32,32)` 也能凑出一样的 shape，却会把像素重新分组，颜色和位置就对不上了。图里偶尔用 `B` 表示 batch，本章代码用 `N`，两者指的是同一维。

## 三个通道分别标准化

`normalize()` 先把像素除以 255，再按通道计算：

$$
x'_{nchw}=\frac{x_{nchw}/255-\mu_c}{\sigma_c}
$$

代码预设了下面这组 CIFAR-100 通道常数，所有 split 都使用它们：

```python
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)
```

均值和标准差会 reshape 成 `(1,3,1,1)`，通过广播作用到所有样本和空间位置；输出仍是 `(N,3,32,32)`，dtype 为 `float32`。标准化并不是把数值压进 0 到 1：低于通道均值的像素会变成负数，高于均值较多的像素也可能大于 1。

这些常数并不是程序对当前 45,000 张训练图片重新计算的结果，改变 `seed` 或 `train_limit` 也不会改变它们。若要在自己的数据上估计标准化参数，应先划分 train/validation，把 train 像素除以 255 后，再计算 `mean(axis=(0,2,3))` 和 `std(axis=(0,2,3))`，并保存给后续推理使用；不要给 validation 和 test 各算一套。这里的轴与后面的 BatchNorm2D 相同，但它只在准备数据时计算一次，BatchNorm 则会处理每一层不断变化的激活值。

## 只有训练 batch 会随机改变

训练循环会做两种增强：外围补 4 格后随机裁回 `32×32`，再以 0.5 的概率水平翻转。验证和测试保持不变。

![本仓库实际采用的裁剪与翻转顺序](assets/data_augmentation.png)

这里需要留意实际调用顺序：

```text
load_cifar100_splits
  -> to_nchw
  -> normalize

train_epoch
  -> padding + random crop
  -> horizontal flip
```

也就是说，本仓库是在标准化之后补 0。此时的 0 对应各通道均值，并不是原始 RGB 的黑色。许多 torchvision 示例会先裁剪、翻转，再标准化，因此仅看函数名字不足以判断 padding 代表什么颜色。

在这份实现里，`random_crop_with_padding()` 为每张图独立抽取左上角，`random_horizontal_flip()` 沿 NCHW 的最后一轴翻转。两者都返回新数组，不改写传入的 batch。

增强只改变图片，标签必须跟着原样保留；shuffle 也要让图片和标签使用同一组索引，否则答案就会错位。

## 一个短 batch 也要保留下来

`iterate_minibatches()` 先生成样本索引，按需打乱，再用同一批索引选择图片和标签。10 条数据、`batch_size=4` 时会产生三个 batch：

```text
4, 4, 2
```

最后两条不会被丢掉。因此统计一轮平均 loss 时，要按每个 batch 的实际样本数加权；后面的 SmallResNet 训练循环采用了这一写法。

下面的例子不下载数据，只检查轴顺序、dtype 和 batch 配对：

```python
import numpy as np
from exercises.block_02_resnet.task_10_image_data_pipeline.data_pipeline import (
    iterate_minibatches,
    normalize,
    to_nchw,
)

images = np.zeros((10, 32, 32, 3), dtype=np.uint8)
labels = np.arange(10, dtype=np.int64)
images = normalize(to_nchw(images), [0.5] * 3, [0.5] * 3)

print(images.shape, images.dtype)
for xb, yb in iterate_minibatches(images, labels, 4, shuffle=False):
    print(xb.shape, yb.tolist())
```

输出为：

```text
(10, 3, 32, 32) float32
(4, 3, 32, 32) [0, 1, 2, 3]
(4, 3, 32, 32) [4, 5, 6, 7]
(2, 3, 32, 32) [8, 9]
```

## 在代码里对应哪些函数

`load_cifar100_splits()` 负责下载后的拆分、NCHW 转换和标准化，`iterate_minibatches()` 只按索引组成 batch，随机裁剪与翻转则由 `train_epoch(..., augment=True)` 调用。把这三个阶段分开看，更容易发现图片在哪一步发生了变化；对应的检查集中在 Block 2 测试中：

```bash
python -m unittest discover -s tests -p 'test_block2.py' -v
```

下一页的 [Conv2D 与 im2col](../task_11_conv2d_im2col/README.md) 直接接收这里得到的 NCHW batch。

参考：[CIFAR-10 与 CIFAR-100 官方说明](https://www.cs.toronto.edu/~kriz/cifar.html)、[CS231n 数据预处理](https://cs231n.github.io/neural-networks-2/)。
