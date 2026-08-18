# Block 2：用 NumPy ResNet 分类 CIFAR-100

Block 1 的 MLP 接收一维特征。图像多了两个值得利用的条件：相邻像素往往相关，同一种局部结构也可能出现在不同位置。卷积用局部连接和参数共享表示这两个条件；残差连接则让较深的网络更容易优化。

本章使用 **CIFAR-100**。它包含 60,000 张 $32\times32$ RGB 图片，共 100 个细类别；每类有 500 张官方训练图片和 100 张官方测试图片。代码从 50,000 张官方训练图片中划出验证集，10,000 张官方测试图片只用于最终评估。数据规模和类别定义可查阅 [CIFAR 官方页面](https://www.cs.toronto.edu/~kriz/cifar.html)。

![从 CIFAR-100 图片到类别分数](../assets/images/block2_overview.png)

这是一套教学实现，不是准确率基准。重点是把数据布局、卷积反向传播、BatchNorm 状态、残差分支和 checkpoint 连成一条可检查的训练链路。

---

## 内容路线

| 部分 | 代码入口 | 核心内容 |
| --- | --- | --- |
| task 10 | `data_pipeline.py` | 隔离的数据划分与 NCHW batch |
| task 11 | `conv2d.py` | `im2col`、`col2im`、`Conv2D` |
| task 12 | `layers.py` | MaxPool、GlobalAvgPool、BatchNorm |
| task 13 | `residual_block.py` | identity / projection 两种 BasicBlock |
| task 14 | `train_resnet.py` | 可运行的 SmallResNet 训练循环 |
| task 15 | `NOTES.md` | 带命令、配置和指标的实验记录 |

每个部分都配有局部核对项，包括 shape、数值梯度、单 batch 拟合和 checkpoint round-trip。它们把异常限定在具体的数据或网络层，比直接调整训练超参数更容易解释。

---

## 数据统一为 NCHW

`torchvision` 中的 CIFAR-100 图片数组是 `NHWC`：

```text
(N, H, W, C) = (N, 32, 32, 3)
```

本章的卷积和池化统一接收 `NCHW`：

```text
(N, C, H, W) = (N, 3, 32, 32)
```

`to_nchw()` 用 `transpose(0, 3, 1, 2)` 调整轴顺序；`normalize()` 再把像素缩放到 $[0,1]$，按 RGB 通道做

$$
x'_{nchw}=\frac{x_{nchw}/255-\mu_c}{\sigma_c}.
$$

默认划分为 45,000 张训练图片、5,000 张验证图片和 10,000 张官方测试图片。训练/验证从官方训练集分层拆分，测试集不参与选模型。训练循环可以做随机裁剪和水平翻转，验证和测试没有随机增强。

详细说明：[task_10 图像数据管线](../exercises/block_02_resnet/task_10_image_data_pipeline/README.md)

---

## 卷积层的计算约定

深度学习库通常不翻转卷积核，实际计算的是 cross-correlation。对输入 `X` 和权重 `W`，本章采用：

```text
X: (N, C_in, H, W)
W: (C_out, C_in, K_h, K_w)
b: (C_out,)
Y: (N, C_out, H_out, W_out)
```

$$
H_{out}=\left\lfloor\frac{H+2P_h-K_h}{S_h}\right\rfloor+1,
\qquad
W_{out}=\left\lfloor\frac{W+2P_w-K_w}{S_w}\right\rfloor+1.
$$

![可逐项核对的互相关例子](../assets/images/conv2d_explained.png)

图中的 $5\times5$ 输入与 $3\times3$ kernel 在 `stride=1, padding=0` 时得到：

$$
\begin{bmatrix}
0&0&0\\
0&3&-2\\
1&0&0
\end{bmatrix}.
$$

这是配图和代码共同使用的验算样例。

### im2col

逐位置循环容易读，却不适合作为矩阵实现。`im2col` 把每个局部窗口放到一行：

```text
X_col: (N * H_out * W_out, C_in * K_h * K_w)
W_col: (C_in * K_h * K_w, C_out)
Y_col: (N * H_out * W_out, C_out)
```

$$
Y_{col}=X_{col}W_{col}+b.
$$

对 $4\times4$ 单通道输入和 $3\times3$ kernel，共有四个合法窗口，所以 `X_col.shape == (4, 9)`。反向时 `col2im` 会把重叠窗口的梯度**累加**回输入位置，它不是普通 reshape。

另一个工程约束是梯度数组的身份。优化器在构造时保存 `dW`、`db` 的引用，因此 `Conv2D.backward()` 使用 `self.dW[...] = ...` 和 `self.db[...] = ...` 原位写入；重新绑定数组会让优化器继续读取旧梯度。

详细说明：[task_11 Conv2D 与 im2col](../exercises/block_02_resnet/task_11_conv2d_im2col/README.md)

---

## 池化与 BatchNorm 的状态

### MaxPool2D

MaxPool 对每个通道独立取窗口最大值。forward 缓存 `argmax`，backward 把该窗口的上游梯度送回对应位置。并列最大值采用 NumPy 的规则：按行优先展开后取第一个最大值。

![MaxPool 前向和四个上游梯度的路由](../assets/images/maxpool.png)

### GlobalAvgPool2D

$$
y_{nc}=\frac{1}{HW}\sum_{h,w}x_{nchw},
\qquad (N,C,H,W)\rightarrow(N,C).
$$

backward 把每个通道的梯度均分到 $H\times W$ 个空间位置。最终的 `SmallResNet` 使用 GlobalAvgPool；它没有使用 MaxPool，空间降采样由 stage 开头的 stride-2 卷积完成。

### BatchNorm2D

训练时，每个通道在 `(N,H,W)` 三个轴上统计均值和方差：

$$
\hat{x}=\frac{x-\mu_B}{\sqrt{\sigma_B^2+\varepsilon}},
\qquad y=\gamma\hat{x}+\beta.
$$

同时更新 `running_mean` 和 `running_var`。`eval()` 改用这两个 buffer，不再读取当前 batch 的统计量。因此完整的 checkpoint 既包含可训练的 `gamma/beta`，也包含运行统计量。

详细说明：[task_12 池化与 BatchNorm](../exercises/block_02_resnet/task_12_pooling_and_bn/README.md)

---

## BasicBlock 的两条分支

本章实现的是 post-activation BasicBlock：

```text
main:     Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN
shortcut: identity 或 Conv1x1 -> BN
output:   ReLU(main + shortcut)
```

写成公式是

$$
y=\operatorname{ReLU}(F(x)+S(x)).
$$

当 `stride == 1` 且输入、输出通道相同时，$S(x)=x$。空间尺寸或通道数改变时，shortcut 用 `Conv1x1 + BN` 对齐 shape。两路输出同形后再逐元素相加。

![包含第二个 BN 和相加后 ReLU 的 BasicBlock](../assets/images/residual_block.png)

反向传播会在加法节点分成两路，再在块的输入处相加。identity shortcut 的 Jacobian 是单位阵；projection shortcut 的 Jacobian 由 `Conv1x1 + BN` 决定。因此残差连接提供了额外的梯度路径，但不构成“梯度一定不会消失”的保证。

详细说明：[task_13 残差块](../exercises/block_02_resnet/task_13_residual_block/README.md)

---

## SmallResNet 的训练边界

`SmallResNet` 的结构为：

```text
Conv3x3 -> BN -> ReLU
stage 1 -> stage 2(stride 2) -> stage 3(stride 2)
GlobalAvgPool -> Linear -> 100 logits
```

类默认配置是 `channels=(16,32,64)`、每个 stage 两个块；task 14 的命令行默认更小，使用 `(8,16,32)` 和每个 stage 一个块。两者都属于教学用的小型网络，不等同于 ImageNet ResNet-18，也不等同于原论文的 CIFAR ResNet-20。

无下载 smoke test：

```bash
python exercises/block_02_resnet/task_14_numpy_resnet_train/train_resnet.py \
  --synthetic --epochs 1 --channels 2 4 8 --blocks 1 1 1
```

输出包含一行训练/验证指标和一行测试指标：

```text
epoch=1 train_loss=... train_acc=... val_loss=... val_acc=...
test_loss=... test_acc=...
```

这里检查的是数值有限、forward/backward/update 已连通，不用合成数据的准确率代表 CIFAR-100 表现。

task 14 的 CIFAR 路径默认限制训练、验证、测试各 500 张，适合检查实现：

```bash
python exercises/block_02_resnet/task_14_numpy_resnet_train/train_resnet.py \
  --epochs 1 --train-limit 500 --val-limit 200 --test-limit 200 \
  --channels 8 16 32 --blocks 1 1 1
```

完整 checkpoint 由 `solutions/block_02_resnet/train_cifar100_solution.py` 保存，内容包括：

- 命名参数；
- BatchNorm running buffers；
- optimizer 数组状态和超参数；
- epoch、config、history；
- 模型的 train/eval 模式。

加载到同配置的新模型后，round-trip 检查会比较 eval logits，以及继续一步训练后的参数更新。

详细说明：[task_14 NumPy ResNet 训练](../exercises/block_02_resnet/task_14_numpy_resnet_train/README.md)

---

## 运行与核对

```bash
python -m unittest discover -s tests -p 'test_block2.py' -v
python tests/test_im2col_figure.py -v
```

各类测试对应的问题范围如下：

1. 数据 shape、标签范围和 split 隔离；
2. `Conv2D`、BatchNorm 的有限差分；
3. optimizer 保存旧梯度引用时的参数更新；
4. BasicBlock 两种 shortcut 的 forward/backward；
5. 小 batch 上的 loss 下降；
6. checkpoint round-trip。

这些测试只覆盖局部实现性质，不代表模型已经在完整 CIFAR-100 上达到某个准确率。[task_15 实验记录](../exercises/block_02_resnet/task_15_experiment_notes/README.md) 给出了训练配置、耗时和结果的记录格式。

## 参考资料

- [CIFAR-10 and CIFAR-100 datasets](https://www.cs.toronto.edu/~kriz/cifar.html)
- [Stanford CS231n: Convolutional Networks](https://cs231n.github.io/convolutional-networks/)
- [Dive into Deep Learning: Convolutional Neural Networks](https://d2l.ai/chapter_convolutional-neural-networks/index.html)
- [Batch Normalization](https://arxiv.org/abs/1502.03167)
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
