# task_12: 池化与 BatchNorm

现在你已经有了卷积.

卷积负责从局部窗口里提特征. 但一个 CNN 只靠卷积还不够.

图片一开始是 $32\times 32$, 如果每一层都保持这个尺寸, 计算量会越来越大. 而且分类任务最后只需要判断“这张图是什么”, 不需要保留每个像素位置的输出.

所以我们需要几个新层:

- MaxPool: 让空间尺寸变小.
- GlobalAvgPool: 把整张特征图汇聚成一个向量.
- BatchNorm: 让训练更稳.

这些层看起来不像卷积那么“主角”, 但 ResNet 训练不起来时, 问题经常就藏在这里.

---

## 一. MaxPool: 只留下局部最强响应

MaxPool 的想法很粗暴.

在一个小窗口里, 只保留最大值.

比如 $2\times 2$ 的窗口:

$$
\begin{bmatrix}
1 & 3 \\
2 & 0
\end{bmatrix}
$$

MaxPool 以后就是:

$$3$$

为什么可以这么做?

因为很多时候, 我们关心的是“某个特征有没有出现”, 而不是它在这个小窗口里的精确位置.

比如一个边缘检测器在左上角强一点还是右下角强一点, 对后面的分类不一定重要. MaxPool 会让模型对小范围平移更不敏感.

如果输入是:

```text
(N, C, H, W)
```

使用 `kernel_size=2, stride=2`, 输出通常会变成:

```text
(N, C, H/2, W/2)
```

通道数不变, 空间尺寸减半.

### MaxPool 的反向传播

MaxPool forward 只保留最大值, backward 也只把梯度传给最大值所在的位置.

刚才的例子里最大值是 3, 如果上游梯度是 10, 那么梯度会变成:

$$
\begin{bmatrix}
0 & 10 \\
0 & 0
\end{bmatrix}
$$

所以 forward 时必须记住每个窗口里最大值的位置. 否则 backward 不知道梯度该传给谁.

这就是 `MaxPool2D` 里最容易漏掉的缓存.

---

## 二. GlobalAvgPool: 分类前的收口

早期 CNN 常常在最后接一大段全连接层.

但这样参数很多, 也容易过拟合.

Global Average Pooling 更简单: 对每个通道的整张特征图求平均.

如果输入是:

```text
(N, C, H, W)
```

输出就是:

```text
(N, C)
```

也就是说, 每个通道最后只留下一个数字.

你可以把它理解成: 这个通道负责检测某种特征, GlobalAvgPool 问的是“这种特征在整张图里平均有多强”.

当前 `GlobalAvgPool2D` 已经给出了 forward 和 backward:

```python
return np.mean(x, axis=(2, 3))
```

反向传播时, 一个通道上的梯度会平均分回所有空间位置:

$$dx = \frac{dout}{H\times W}$$

---

## 三. BatchNorm: 让每层看到的分布别乱飘

Block 1 里已经讲过归一化.

到了 CNN 里, BatchNorm 很常见, ResNet 里几乎每个卷积后面都会接 BN.

它解决的问题是: 每一层的输入分布在训练过程中会变.

参数一更新, 前一层输出变了, 后一层看到的数据分布也变了. 如果分布飘得太厉害, 训练就会抖, 学习率也不敢设大.

BatchNorm 的做法是, 对一个 batch 内每个通道分别计算均值和方差:

$$\mu_B = \mathrm{mean}(x)$$

$$\sigma_B^2 = \mathrm{var}(x)$$

然后标准化:

$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \varepsilon}}$$

再加上可学习的缩放和平移:

$$y = \gamma\hat{x} + \beta$$

对 2D 图像来说, 每个通道的均值和方差会在 `N,H,W` 三个维度上统计. 所以 `BatchNorm2D` 里的参数形状是:

```text
gamma.shape = (1, C, 1, 1)
beta.shape  = (1, C, 1, 1)
```

这样它们可以广播到整张特征图.

---

## 四. 训练和推理为什么不一样?

BatchNorm 最烦人的地方在这里.

训练时, 它使用当前 batch 的均值和方差.

但推理时, 你可能一次只输入一张图片. 如果还用当前 batch 统计量, 那统计会很不稳定.

所以训练时 BN 会顺手维护两个滑动平均:

```text
running_mean
running_var
```

推理时就用这两个值.

这也是为什么 `BatchNorm2D` 里有:

```python
train()
eval()
```

如果训练时忘了 `train()`, 或验证时忘了 `eval()`, 模型结果会很奇怪.

这类 bug 很常见, 而且不一定报错.

---

## 五. BatchNorm backward 怎么想?

BatchNorm 的 forward 是几步连起来的:

```text
mean -> var -> normalize -> scale and shift
```

反向传播就是把这几步倒着拆开.

最容易先写对的是:

$$d\gamma = \sum dout \cdot \hat{x}$$

$$d\beta = \sum dout$$

难一点的是 $dx$.

你可以先不要死背公式, 而是把计算图写出来:

```text
x -> mean
x -> x - mean
x - mean -> var
var -> std_inv
x_hat = (x - mean) * std_inv
y = gamma * x_hat + beta
```

然后按链式法则往回推.

如果实在卡住, 先用数值梯度检查. BN 的 backward 手推有点烦, 但只要 forward 的缓存清楚, 它不是玄学.

---

## 六. 你要完成什么?

请完成 `layers.py` 里的:

```text
MaxPool2D.forward
MaxPool2D.backward
BatchNorm2D.backward
```

`GlobalAvgPool2D` 已经给了一个可用版本, 你可以读一下它的 forward 和 backward, 这对理解“空间维汇聚”很有帮助.

建议测试顺序:

1. 先用一个很小的数组测试 MaxPool forward.
2. 再测试 MaxPool backward, 看梯度是不是只回到最大值位置.
3. 测试 BatchNorm forward, 看每个通道标准化后均值是否接近 0、方差是否接近 1.
4. 最后做 BatchNorm backward 的数值梯度检查.

下一关我们把卷积、BN、ReLU 组合起来, 做 ResNet 里最重要的残差块.
