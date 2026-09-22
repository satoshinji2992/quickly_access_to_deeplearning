# 两条路径怎样组成残差块

[ResNet 论文的图 1](https://arxiv.org/abs/1512.03385)在 CIFAR-10 上比较过 20 层和 56 层的普通网络。更深的那一个连训练误差都更高，说明问题不只是验证集上的过拟合：增加的层没有顺利学成原来较浅网络也能表示的函数。

[`residual_block.py`](./residual_block.py) 给主分支旁边加了一条 shortcut。先看两路在哪里分开、在哪里相加，再读 backward。

![BasicBlock 的主分支和 shortcut](assets/residual_block.png)

图中省略了 batch，用 `H×W×C` 描述特征图的高、宽和通道数；代码里的数组仍按 `(N,C,H,W)` 排列。`H/s`、`W/s` 表示图中尺寸可整除的示例，遇到奇数高、宽时，要用前面的卷积输出公式计算。例如输入高为 5、stride 为 2 时，两条分支的输出高都是 3。

## 先把加法前后的量分清

主分支记为 `F(x)`，shortcut 记为 `S(x)`。两路相加得到 `z`，最后经过 ReLU 得到块输出 `y`：

$$
z=F(x)+S(x)
$$

$$
y=\operatorname{ReLU}(z)
$$

shape 不变时，`S(x)=x`。假如希望加法点得到某个值 `z_*`，主分支学的就是相对于输入还差多少：

$$
F(x)=z_*-x
$$

若 `z_*` 和 `x` 已经很接近，`F(x)` 只需要学一个小修正。这里的“残差”不是分类错误或 loss，而是加法前主分支要补上的变化。块的最终输出仍是 `ReLU(F(x)+x)`，不要把加法点 `z` 和 ReLU 之后的 `y` 混为一个量。

## 代码中的 BasicBlock

主分支按下面的顺序计算：

```text
Conv3x3(stride=s)
-> BatchNorm
-> ReLU
-> Conv3x3(stride=1)
-> BatchNorm
```

shortcut 有两种：

```text
shape 不变  identity
shape 改变  Conv1x1(stride=s) -> BatchNorm
```

两路相加后还有一次 ReLU：

```text
main + shortcut -> ReLU
```

第二个卷积后没有立即做 ReLU，而是先经过第二个 BatchNorm，再和 shortcut 相加；漏掉 `bn2` 或末尾 ReLU，都会变成另一种网络。

<div class="widget-mount" data-widget="residual-block"></div>

## identity 路径不改 shape

```text
BasicBlock(in_channels=16, out_channels=16, stride=1)
input   (N,16,32,32)
output  (N,16,32,32)
```

主分支的两个 `3×3` 卷积都使用 padding 1，空间大小不变；shortcut 直接使用输入。相加时两边都是 `(N,16,32,32)`。

## projection 路径同时对齐通道和空间

```text
BasicBlock(in_channels=16, out_channels=32, stride=2)
input   (N,16,32,32)
output  (N,32,16,16)
```

主分支的第一个 `3×3` 卷积用 stride 2，把高、宽减半并产生 32 个通道。shortcut 已经不能原样相加，于是也用 `1×1` 卷积完成通道变化和降采样，再接 BatchNorm。代码通过下面的条件决定是否需要这次投影：

```python
self.needs_projection = self.stride != 1 or self.in_channels != self.out_channels
```

`1×1` 说的是空间范围，不是只有一个权重。对于这里的投影，每个输出位置仍会读取输入同一位置的全部 16 个通道，一组 16 个权重产生一个新通道，32 组便产生 32 个通道。它相当于在每个位置共用一个 `Linear(16,32)`；stride 2 再决定只读取间隔为 2 的空间位置。这样就能对齐两路 shape，而不用在 shortcut 上再做一次 `3×3` 的邻域混合。

forward 在相加前还会比较两路 shape。不一致时立即报错，比等待 NumPy 广播出一个看似能运行的结果更安全。

## forward 按什么顺序留下缓存

把代码缩成数据流，就是：

```text
main = conv1(x)
main = bn1(main)
main = relu1(main)
main = conv2(main)
main = bn2(main)

shortcut = x
# 或 shortcut = proj_bn(proj_conv(x))

y = relu2(main + shortcut)
```

每个子层保存自己 backward 所需的输入或中间量。下面两组小输入可以直接看到 identity 与 projection 的 shape：

```python
import numpy as np
from exercises.block_02_resnet.task_13_residual_block.residual_block import BasicBlock

rng = np.random.default_rng(0)

identity = BasicBlock(4, 4, stride=1)
x1 = rng.normal(size=(2, 4, 8, 8))
y1 = identity.forward(x1)
dx1 = identity.backward(np.ones_like(y1))
print(x1.shape, y1.shape, dx1.shape)

projection = BasicBlock(4, 8, stride=2)
x2 = rng.normal(size=(2, 4, 8, 8))
y2 = projection.forward(x2)
dx2 = projection.backward(np.ones_like(y2))
print(x2.shape, y2.shape, dx2.shape)
```

输出为：

```text
(2,4,8,8) (2,4,8,8) (2,4,8,8)
(2,4,8,8) (2,8,4,4) (2,4,8,8)
```

## 加法节点怎样把梯度分成两份

先经过末尾 ReLU：

```text
dadded = relu2.backward(dout)
```

因为 `z = main + shortcut`，两条分支都收到同一个 `dadded`。主分支逆着 forward 返回：

```text
bn2 -> conv2 -> relu1 -> bn1 -> conv1
```

identity shortcut 直接令 `dshortcut=dadded`；projection shortcut 则经过 `proj_bn -> proj_conv`。两边都回到输入后再相加：

```text
dx = dmain + dshortcut
```

![主分支与 shortcut 都对输入梯度有贡献](assets/residual_gradient_path.png)

用标量看这一步更直观。暂时假设末尾 ReLU 正在正半轴，主分支局部变化率为 0.1，shortcut 为恒等映射。若上游梯度为 2：

```text
主分支贡献    2×0.1 = 0.2
shortcut 贡献 2×1   = 2.0
合计                  2.2
```

这说明 shortcut 提供了另一项，不说明梯度永远不衰减。ReLU 可能截断梯度，主分支的导数也可能和 shortcut 抵消。

## 模式和状态要沿两条分支传下去

一个块里有两到三个 Conv2D，也有两到三个 BatchNorm。`parameters()` 收集卷积权重和 `gamma/beta`，`named_buffers()` 收集各个 BatchNorm 的 `running_mean/running_var`。

`block.train()` 和 `block.eval()` 还要把模式递归传给所有子层。只修改块自己的 `training` 标记没有用；BatchNorm 收不到通知，验证时仍会读取当前 batch 的统计量。

数值与 shape 检查集中在：

```bash
python -m unittest discover -s tests -p 'test_block2.py' -v
```

下一页把多个块接成一个可训练的 [SmallResNet](../task_14_numpy_resnet_train/README.md)。

参考：[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)、[Dive into Deep Learning：残差网络](https://d2l.ai/chapter_convolutional-modern/resnet.html)。
