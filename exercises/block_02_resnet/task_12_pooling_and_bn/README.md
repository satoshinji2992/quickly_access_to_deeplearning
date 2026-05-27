# task_12：池化与 BatchNorm

前一个任务里，我们已经让卷积层学会了“看局部”。它可以从图片里抓出边缘、颜色变化、局部纹理。可是真正把 CNN 训练起来时，你很快会遇到两个新的麻烦：

第一，特征图一直保持很大的空间尺寸，计算量会越来越重；第二，每一层的输出分布都在变，后面的层像是在追着一个不断移动的靶子训练。

这一节要补上三个很常见、也很容易被初学者低估的层：`MaxPool2D`、`GlobalAvgPool2D` 和 `BatchNorm2D`。

它们不负责“发明新类别”，但负责让网络更像一个能训练的系统。

## 1. 这个任务解决什么问题

卷积层输出的是四维张量：

```text
(N, C, H, W)
```

`N` 是 batch size，`C` 是通道数，`H/W` 是空间尺寸。比如输入 CIFAR-100 图片是 `(N, 3, 32, 32)`，经过一个 3x3 卷积后，可能变成 `(N, 16, 32, 32)`。如果我们继续堆很多层，空间尺寸一直不变，计算会很重。

但分类任务不需要最后还保留“每个像素”的判断。我们最后只需要回答：这张图更像苹果、公交车、松鼠，还是别的类别？

于是出现了池化：

- `MaxPool2D`：在一个小窗口里取最大值，把空间尺寸缩小。
- `GlobalAvgPool2D`：在最后把每个通道的整张特征图取平均，得到一个向量。

还有一个训练问题：卷积输出的数值分布会随着参数更新不断变化。前一层稍微变一点，后一层收到的输入分布就变了。这会让训练变得抖。

于是出现了 BatchNorm：

- 在训练时，用当前 batch 估计均值和方差。
- 把特征标准化到比较稳定的尺度。
- 再用可学习的 `gamma` 和 `beta` 让模型自己决定是否放大、缩小、平移。

这三个层合起来，会让后面的 ResNet 更容易训练。

## 2. 如何让模型学会这个？

先看池化。

如果卷积层已经在某个区域里检测到了“这里有一条边”或者“这里有一块有用纹理”，那我们未必关心它精确出现在 8 行 9 列，还是 9 行 10 列。对分类来说，很多时候只需要知道“这一小片区域里出现过这个特征”。

`MaxPool2D` 做的事就很粗暴：一个窗口里谁最大，谁留下。

```text
[[1, 2],
 [5, 3]]

max = 5
```

它的直觉是：激活越大，说明这个特征越强。留下最强的证据，丢掉一部分位置细节，模型反而更稳。

再看 BatchNorm。

你在任务一做线性回归时，已经遇到过标准化：如果收入、年龄、房价这些量纲差太多，梯度下降会很难走。CNN 里也类似。不同通道的激活值如果尺度差异太大，后面的层学习会变别扭。

BatchNorm 的想法是：每个通道单独标准化。

不是对整张图片所有数混在一起算一个均值，而是对每个通道，把 batch 和空间位置都拿来统计：

```text
mean[c] = 第 c 个通道在 N、H、W 上的平均值
var[c]  = 第 c 个通道在 N、H、W 上的方差
```

这样做之后，一个通道负责的特征会更稳定。模型不必每一层都重新适应前面层的数值漂移。

最后看 `GlobalAvgPool2D`。

传统 CNN 末尾常接全连接层。可是如果特征图很大，全连接参数会很多。全局平均池化更简单：每个通道取一个平均值。

```text
(N, C, H, W) -> (N, C)
```

你可以把它理解成：每个通道已经学成了一个“检测器”，最后问一句“这个检测器在整张图上平均有多强”。

## 3. 需要引入哪些数学对象

`MaxPool2D` 的输出尺寸和卷积很像。输入是 `(N, C, H, W)`，池化窗口是 `K`，步幅是 `S`：

```text
H_out = (H - K) // S + 1
W_out = (W - K) // S + 1
```

输出：

```text
y[n, c, i, j] = max(x[n, c, i*S : i*S+K, j*S : j*S+K])
```

反向传播时要注意：梯度只给到当时最大的那个位置。

如果前向里这个窗口最大值来自左下角，那么反向时 `dout` 也只回到左下角。其他位置前向没有影响输出，反向梯度就是 0。

`GlobalAvgPool2D` 更简单：

```text
y[n, c] = mean(x[n, c, :, :])
```

反向传播时，输出的一个梯度要平均分回 `H * W` 个位置：

```text
dx[n, c, h, w] = dout[n, c] / (H * W)
```

`BatchNorm2D` 的前向公式是：

```text
mu = mean(x, axis=(0, 2, 3), keepdims=True)
var = var(x, axis=(0, 2, 3), keepdims=True)

x_hat = (x - mu) / sqrt(var + eps)
y = gamma * x_hat + beta
```

其中：

- `gamma` 的形状是 `(1, C, 1, 1)`。
- `beta` 的形状是 `(1, C, 1, 1)`。
- `eps` 是为了避免除以 0。
- `running_mean` 和 `running_var` 是给 eval 阶段用的移动平均。

训练时用当前 batch 的均值方差；验证和测试时不能再依赖当前 batch，所以要用训练阶段积累的 running statistics。

这是 BatchNorm 最容易写错的地方。

## 4. 对应代码该怎么组织

这一节主要改 [layers.py](/root/code/python/quickly_access_to_deeplearning/exercises/block_02_resnet/task_12_pooling_and_bn/layers.py)。

你会看到三个类：

```python
class MaxPool2D:
    def forward(self, x):
        ...

    def backward(self, dout):
        ...


class GlobalAvgPool2D:
    def forward(self, x):
        ...

    def backward(self, dout):
        ...


class BatchNorm2D:
    def forward(self, x):
        ...

    def backward(self, dout):
        ...
```

`GlobalAvgPool2D` 已经基本写好，可以作为一个很好的对照：前向保存输入 shape，反向用这个 shape 把梯度分回去。

`MaxPool2D` 也需要保存前向信息。至少要保存：

- 输入 `x` 的 shape。
- 每个池化窗口最大值的位置，或者保存一个 mask。

`BatchNorm2D` 需要保存：

- `x_centered`
- `std_inv`
- `x_hat`
- 训练时使用的 batch 方差
- 或者任何你推反向传播需要的中间量

如果你发现 backward 写不出来，不要先怀疑自己数学不行。BatchNorm 的反向确实比前面几层复杂。可以先写清楚每个中间变量，再一行一行反推。

## 5. 逐步实现顺序

建议按这个顺序做。

第一步，先实现 `MaxPool2D.forward`。

拿一个很小的输入手算：

```python
x = np.array([[[[1, 2, 3, 4],
                [5, 6, 7, 8],
                [1, 1, 2, 2],
                [3, 4, 5, 6]]]])
```

如果 `kernel_size=2, stride=2`，输出应该是：

```text
[[[[6, 8],
   [4, 6]]]]
```

先别急着写 batch 版本。你可以先在脑子里确认单张图、单通道怎么取，再扩展到 `N` 和 `C`。

第二步，实现 `MaxPool2D.backward`。

假设上面输出位置 `(0,0)` 的最大值是 6，它来自原窗口里的右下角。那么 `dout` 对应位置的梯度，只能回到原来的右下角。

如果一个窗口里有多个相同最大值，有两种常见处理：

- 只把梯度给 `np.argmax` 找到的第一个最大值。
- 平均分给所有最大值。

为了实现简单，建议用第一种。保持前向和反向一致就行。

第三步，读懂 `GlobalAvgPool2D`。

它已经实现了：

```python
return np.mean(x, axis=(2, 3))
```

你要重点看它的 backward 为什么是：

```python
dout[:, :, None, None] * np.ones((n, c, h, w)) / (h * w)
```

这句代码就是把 `(N, C)` 的梯度扩成 `(N, C, H, W)`，再平均分给每个位置。

第四步，补 `BatchNorm2D.backward`。

最稳的写法是沿着计算图拆：

```text
x -> mean
x -> var
x -> x_centered
x_centered + var -> x_hat
x_hat -> gamma/beta -> y
```

可以先求：

```python
dgamma = np.sum(dout * self.x_hat, axis=(0, 2, 3), keepdims=True)
dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)
```

然后再求 `dx`。

第五步，做数值梯度检查。

不要用大张量检查。比如：

```text
x.shape = (2, 3, 4, 4)
```

就够了。数值梯度本来就慢，小输入更容易定位问题。

## 6. 常见错误和调试方法

最常见的问题是 axis 写错。

BatchNorm2D 是对每个通道统计，所以均值方差的 axis 应该是 `(0, 2, 3)`，不是 `(0, 1, 2, 3)`。如果把通道也平均掉，所有通道会被压成同一个尺度，信息会被破坏。

第二个问题是 broadcasting 形状不对。

`gamma` 和 `beta` 应该是 `(1, C, 1, 1)`。这样才能自动广播到 `(N, C, H, W)`。如果你把它写成 `(C,)`，有时 NumPy 会广播失败，有时会在你没意识到的维度上广播。

第三个问题是 eval 阶段还在用当前 batch 的均值方差。

训练时：

```python
bn.train()
```

验证时：

```python
bn.eval()
```

如果验证时还用当前 batch 的统计量，结果会不稳定，而且 batch size 变小时尤其明显。

第四个问题是 MaxPool backward 把梯度传给了整个窗口。

平均池化才是平均分梯度。最大池化只给最大值位置。可以打印一个小窗口的 mask 看看：

```text
window:
[[1, 2],
 [5, 3]]

mask:
[[0, 0],
 [1, 0]]
```

第五个问题是池化输出尺寸算错。

如果输入是 32，kernel 是 2，stride 是 2，那么输出是 16；如果 stride 写成 1，输出会变成 31。这个错误会一路传到后面的残差块，最后表现成“加法 shape 对不上”。

## 7. 必做实验

做完代码后，不要只跑一次“没报错”。这一节至少做下面几个实验。

1. 用手写小矩阵测试 `MaxPool2D.forward`。

   输入固定成 `(1, 1, 4, 4)`，手算输出，再和代码结果对比。

2. 检查 `MaxPool2D.backward` 的稀疏性。

   传入全 1 的 `dout`，观察 `dx`。每个池化窗口应该只有一个位置拿到梯度。

3. 检查 `GlobalAvgPool2D.backward`。

   如果输入是 `(2, 3, 4, 4)`，`dout` 是全 1，那么每个位置的梯度应该是 `1 / 16`。

4. 比较 BatchNorm train/eval。

   先用几批随机数据调用 `forward`，让 `running_mean/running_var` 更新。然后切到 `eval()`，看输出是否使用 running statistics。

5. 做 BatchNorm 数值梯度检查。

   分别检查 `dx`、`dgamma`、`dbeta`。如果只检查输出 shape，很容易漏掉反向传播错误。

6. 改变 batch size。

   试试 batch size 为 2、8、32 时，BatchNorm 的输出均值方差是否稳定。你会看到 batch 太小时统计量会更抖。

## 8. 完成后应该能回答的问题

完成这一节后，你应该能回答：

- 为什么分类模型不需要最后保留完整的空间网格？
- MaxPool 和 GlobalAvgPool 分别在网络的什么位置更常见？
- MaxPool backward 为什么不是把梯度分给窗口里的所有元素？
- BatchNorm2D 为什么按通道统计，而不是对所有元素一起统计？
- `gamma` 和 `beta` 为什么是可学习参数？
- 为什么训练和验证时 BatchNorm 的行为不同？
- running mean / running var 是怎么更新的？
- 如果 BatchNorm 的 axis 写错，会出现什么现象？
- 为什么这些层本身不负责分类，却能明显影响训练效果？

如果这些问题答得清楚，你已经开始从“会写层”进入“知道层为什么存在”。

## 9. 下一任务为什么自然出现

现在你已经有了卷积、池化、BatchNorm 和全局平均池化。理论上，我们可以开始堆一个 CNN。

但直接堆深层 CNN 会遇到一个奇怪的问题：层数变多后，模型不一定更好，甚至训练误差都可能变差。

这不是因为深模型表达能力不够。按理说，深模型至少可以学成浅模型的样子。问题在于优化：梯度要穿过很多层，信息也要一层一层被改写，训练会越来越难。

ResNet 的残差连接就是为了解决这个问题。下一节我们会写 `BasicBlock`，让网络学会：

```text
输出 = 卷积分支学到的修正量 + 原来的输入
```

这会让深层网络多一条更直接的信息通路。
