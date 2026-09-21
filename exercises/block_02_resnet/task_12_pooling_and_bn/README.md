# BatchNorm、全局平均池化与 MaxPool

`SmallResNet` 的卷积后面会接 BatchNorm，最后再用 Global Average Pool 把空间维收掉。[`layers.py`](./layers.py) 还实现了 MaxPool，方便对照另一种常见的降采样方式；不过当前模型的中途降采样由 stride-2 卷积完成，所以下面先沿着实际 forward 的顺序看。

## 卷积后的一个通道怎样做 BatchNorm

输入 shape 是 `(N,C,H,W)`。BatchNorm2D 不把所有数字混在一起，而是让每个通道分别统计；对通道 `c`，均值和方差覆盖样本、高度和宽度：

$$
\mu_c=\frac{1}{NHW}\sum_{n,h,w}x_{n,c,h,w}
$$

$$
\sigma_c^2=\frac{1}{NHW}\sum_{n,h,w}(x_{n,c,h,w}-\mu_c)^2
$$

先看某个通道只有 `[1,3]` 两个数的情况。均值是 2，方差是 1；暂时忽略很小的 `ε`，标准化结果就是 `[-1,1]`。

真实层随后还会乘可学习的 `γ`、加可学习的 `β`：

$$
\hat{x}=\frac{x-\mu_c}{\sqrt{\sigma_c^2+\varepsilon}}
$$

$$
y=\gamma\hat{x}+\beta
$$

`gamma` 和 `beta` 的 shape 都是 `(1,C,1,1)`，通过广播作用到该通道的所有位置。输出经过这一步后不必继续保持零均值、单位方差。

![BatchNorm2D 的统计轴、训练状态和运行统计量](assets/batchnorm.png)

## 为什么训练和评估要用两组统计量

训练时，一个 mini-batch 已经摆在内存里，可以直接用它的均值和方差。层还会把这些数缓慢记入 `running_mean` 与 `running_var`：

$$
\operatorname{running\_mean}
\leftarrow m\operatorname{running\_mean}+(1-m)\mu_B
$$

$$
\operatorname{running\_var}
\leftarrow m\operatorname{running\_var}+(1-m)\sigma_B^2
$$

本实现取 `m=0.9`，也就是旧统计量占 90%。有些框架把 `momentum` 定义成新统计量的权重，名字相同但公式相反；对照不同实现时，最好直接看公式。

评估阶段改用已经积累的 running statistics。这样一张图片的预测就不会因为恰好和哪几张图片放在同一个 batch 而变化：

```text
model.train()  当前 batch 的 mean/var；更新 running buffers
model.eval()   running_mean/running_var；不再更新 buffers
```

`gamma` 和 `beta` 是参数，要由优化器更新；`running_mean` 和 `running_var` 是 buffer，不求梯度，但会影响评估输出。因此 checkpoint 必须同时保存两类状态。

### 反向传播沿哪些轴归约

参数梯度沿 `(N,H,W)` 求和：

$$
d\beta=\sum_{n,h,w}dY
$$

$$
d\gamma=\sum_{n,h,w}dY\odot\hat{X}
$$

训练模式下，输入还通过 batch 均值和方差间接影响其他样本。令 `M=NHW`、`dX_hat=dY*gamma`，合并后的公式为：

$$
dX=\frac{1}{\sqrt{\sigma_B^2+\varepsilon}}
\left(
d\hat{X}
-\frac{1}{M}\sum d\hat{X}
-\frac{\hat{X}}{M}\sum(d\hat{X}\odot\hat{X})
\right)
$$

这三项分别来自直接标准化路径、均值路径和方差路径。在代码里，它们都沿 `(N,H,W)` 归约；有限差分测试用来检查实现是否遗漏其中一项。评估模式则把 running statistics 当作常数，输入梯度只剩一次缩放：

$$
dX=dY\odot\gamma/\sqrt{\operatorname{running\_var}+\varepsilon}
$$

`backward()` 会用 `dgamma[...] = ...` 和 `dbeta[...] = ...` 原位写入参数梯度。训练模式的 forward 也会原位更新两个 running buffer；评估模式只读取它们，不再写入。原位更新保留了数组身份，优化器和 checkpoint 代码中已经保存的引用不会失效。

## 网络末尾为什么只留每个通道的平均值

经过最后一个残差 stage 后，CIFAR-100 教学配置得到 `(N,32,8,8)`。Global Average Pool 对每个通道的 `8×8` 网格分别求平均：

$$
y_{n,c}=\frac{1}{HW}\sum_{h,w}x_{n,c,h,w}
$$

shape 因而变成：

```text
(N,32,8,8) -> (N,32)
```

![三个通道分别求空间均值](assets/globalavgpool.png)

如果某个通道只有四个数 `[1,2,3,6]`，输出就是 3；假设这个输出收到梯度 2，四个输入便各收到 `2/4=0.5`。由此也能看出，`GlobalAvgPool2D` 只需记住输入 shape，本身没有参数或运行 buffer。

## MaxPool 是另一种局部缩小方式

MaxPool 不在当前 `SmallResNet` 的 forward 路径中，但许多 CNN 会使用它。每个窗口只留下最大值，输出通道数不变：

$$
H_{out}=\left\lfloor\frac{H-K_h}{S_h}\right\rfloor+1
$$

$$
W_{out}=\left\lfloor\frac{W-K_w}{S_w}\right\rfloor+1
$$

![2×2 MaxPool 的四个窗口与四份上游梯度](assets/maxpool.png)

图中 `kernel_size=2, stride=2`，四个窗口产生四个输出，所以 backward 也有四份上游梯度，每一份都只回到对应窗口前向时的最大值位置。如果最大值并列，还要约定由谁接收梯度；本实现沿用 NumPy `argmax`，窗口按行展开后由第一处最大值获胜。例如：

$$
\begin{bmatrix}
6&6\\
4&6
\end{bmatrix}
$$

左上角的 6 收到整份梯度。若池化窗口互相重叠，同一输入位置还可能从多个窗口获胜；`np.add.at` 会把这些梯度相加，而不是让后写入的值覆盖前一个。

<div class="widget-mount" data-widget="pool-sweep"></div>

## 对照代码时看什么

`BatchNorm2D.forward()` 的分支由 `self.training` 决定；`GlobalAvgPool2D.backward()` 把一个通道的梯度均分回空间位置；`MaxPool2D.forward()` 保存 argmax，backward 再按索引写回。三类层都实现相同的 `forward/backward/train/eval` 外壳，模型才能统一切换状态。对应的数值检查在：

```bash
python -m unittest discover -s tests -p 'test_block2.py' -v
```

下一页的 [BasicBlock](../task_13_residual_block/README.md) 会把 Conv2D、BatchNorm 和 ReLU 接成两条分支。

参考：[Dive into Deep Learning：Batch Normalization](https://d2l.ai/chapter_convolutional-modern/batch-norm.html)、[Batch Normalization 论文](https://arxiv.org/abs/1502.03167)。
