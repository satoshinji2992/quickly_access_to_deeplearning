# 第四节：用 MLP 识别 MNIST

![MNIST MLP](assets/mnist_mlp.png)

前面三节已经把训练流程拆成了几块：

```text
forward → loss → backward → optimizer.step()
```

输入换成手写数字图片后，组件之间的调用顺序不变。`starter.py` 提供两个入口：默认的 `reference` 使用根目录下已填好的 `common/my_dl_lib.py`，`task02` 则直接加载上一节的 `my_dl_lib.py`。

两边共用同一段 MNIST 训练循环，差别只在 Linear、ReLU、CrossEntropyLoss、Sequential 和 Momentum 来自哪个文件。`reference` 模式把数据与训练脚本单独拿出来核对，`task02` 模式则把上一节的库也纳入这条路径；若后者抛出 `NotImplementedError`，报错位置就是还没有接通的组件。

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

10 个 logits 对应数字 0 到 9。脚本将整数标签转成 one-hot；交叉熵计算损失，`argmax` 给出预测类别。圆形分类时最后有 2 个分数，到了 MNIST 只是改成 10 个，softmax 和交叉熵的写法没有变。

## 数据划分

脚本优先下载 MNIST 的原始训练集，再按类别分层抽取 5000 张训练图片和 1000 张验证图片。两组索引没有交集，验证数据只参与 `forward`，不会进入 `backward` 或参数更新。

如果环境中没有 `torchvision`，脚本会改用 $8\times8$ 的 `sklearn.datasets.load_digits`，把图片居中填充到 $28\times28$，并按相同比例缩小训练集和验证集。这个备用数据集适合检查代码流程，但结果不能当作 MNIST 指标来比较。

默认使用填好的参考库：

```bash
cd exercises/block_01_basics/task_03_mnist_mlp
python starter.py
```

直接换成上一节的实现，不需要复制或改名文件：

```bash
python starter.py --library task02
```

两个命令默认都使用随机种子 0；要比较另一组初始权重，可以传入 `--seed 1` 之类的值。

日志开头会打印实际划分，随后每轮同时给出训练与验证指标：

```text
library: reference
data split: train=5000, val=1000 (stratified, disjoint)
epoch=01 train_loss=... train_acc=... val_loss=... val_acc=...
```

当前代码每轮用全部训练图片做一次梯度更新，没有 mini-batch 循环，所以它的收敛比常见的框架示例慢。固定默认种子后，真正的 MNIST 路径会显示 `train=5000, val=1000`，20 轮后的 `val_acc` 约为 `0.65`；这是用来核对这份全批量教学脚本的数值，不是 MNIST 的模型水平上限。

如果看到 `using sklearn digits as a small fallback`，实际划分会变成 `train=1497, val=300`，默认运行的末轮 `val_acc` 约为 `0.70`。两条路径中，`train_loss` 都应该总体下降，而且不出现 `nan` 或 `inf`；两个数据集的准确率不宜直接横向比较。

多跑一些轮数后，如果 `train_acc` 还在上升，`val_acc` 却开始停滞或下降，训练集与验证集之间的分叉才开始表现为过拟合。

## MLP 在图片上的局限

flatten 没有删掉像素值，却把 $28\times28$ 的坐标系改成了一条长度 784 的固定顺序。水平相邻的像素通常在向量中相邻，竖直相邻的像素却相隔 28 个位置，行尾和下一行行首反而会被挨在一起。

更重要的是，全连接层对所有输入位置使用各自的权重，并不知道“空间上相邻”有特殊意义。一个笔画平移几格后，MLP 往往要在新位置重新学相似的模式。

改变隐藏维度会改变模型容量，改学习率则改变每次参数更新的步长。这两件事对日志的影响不一样，单看一次“最终准确率更高”，很难分清究竟是哪种变化起了作用。

如果对所有图片使用同一个固定像素置换，MLP 仍然有机会拟合新的顺序，因为它本来就没有假定哪些位置应当相邻。卷积恰好相反：它把局部邻域写进了结构，乱掉像素位置就会破坏这个先验。下一章会保留图片的二维布局，用卷积核处理局部区域，并在不同位置共享参数。
