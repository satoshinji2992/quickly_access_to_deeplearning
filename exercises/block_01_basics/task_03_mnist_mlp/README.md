# 第四节：用 MLP 识别 MNIST

![MNIST MLP](assets/mnist_mlp.png)

前面三节已经把训练流程拆成了几块：

```text
forward → loss → backward → optimizer.step()
```

这一节沿用同一训练流程，输入换成了手写数字图片。

## 从图片到 logits

MNIST 的每张图片是 $28\times28$ 的灰度图。`starter.py` 将像素缩放到 $[0,1]$，再把每张图片展开：

$$
(B,28,28)\longrightarrow(B,784)
$$

当前 MLP 的 shape 是：

```text
images:  (B, 784)
W1:      (784, 128)
hidden:  (B, 128)
W2:      (128, 10)
logits:  (B, 10)
targets: (B, 10)
```

10 个 logits 对应数字 0 到 9。脚本将整数标签转成 one-hot；交叉熵计算损失，`argmax` 给出预测类别。这与圆形分类的双输出 softmax 完全同形，只是类别数从 2 变成了 10。

## 数据划分

脚本优先下载 MNIST 的原始训练集，再按类别分层抽取 5000 张训练图片和 1000 张验证图片。两组索引没有交集，验证数据只参与 `forward`，不会进入 `backward` 或参数更新。

如果环境中没有 `torchvision`，脚本会使用 `sklearn.datasets.load_digits`。这个备用数据集是 $8\times8$，代码把它居中填充到 $28\times28$，并按相同比例缩小训练集和验证集。它适合检查代码流程，但结果不能当作 MNIST 指标来比较。

运行：

```bash
cd exercises/block_01_basics/task_03_mnist_mlp
python starter.py
```

日志开头会打印实际划分，随后每轮同时给出训练与验证指标：

```text
data split: train=5000, val=1000 (stratified, disjoint)
epoch=01 train_loss=... train_acc=... val_loss=... val_acc=...
```

当前代码每轮用全部 5000 张图片做一次梯度更新。这里没有 mini-batch 循环，所以速度和收敛方式与常见框架训练略有不同。

运行时可以观察到：

- `train_loss` 总体下降；
- `val_acc` 明显高于随机猜测的 10%；
- 验证集没有进入反向传播；
- `train_acc` 持续上升而 `val_acc` 停滞或下降，是常见的过拟合信号。

## MLP 在图片上的局限

flatten 让全连接层可以接收图片，却丢掉了显式的二维结构。相邻像素在向量里仍恰好相邻，但模型不知道“邻近”具有特殊意义，也不会自动让不同位置共享同一组局部特征参数。一个笔画平移几格后，MLP 往往要重新学习相似模式。

下面几类改动分别影响容量、优化和输入结构：

- 隐藏维度从 `128` 变为 `64` 或 `256`，改变模型容量；
- 学习率从 `0.1` 变为 `0.05` 或 `0.2`，改变参数更新的步长；
- 训练轮数增加时，训练曲线与验证曲线可能逐渐分开；
- 所有图片使用同一个固定像素置换时，MLP 仍可以拟合新的输入顺序，而卷积的局部结构先验会被破坏。

下一章会保留图片的二维布局，用卷积核处理局部区域，并在不同位置共享参数。
