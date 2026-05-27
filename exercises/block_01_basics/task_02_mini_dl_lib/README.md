# task_02：整理你的小型深度学习库

## 1. 这个任务解决什么问题

如果你完成了任务二，你现在应该有一个能跑的 MLP。

但它大概率长这样：`Model.py` 里既有权重，又有 ReLU，又有 Softmax，又有交叉熵，又有参数更新。它能跑，但你一改东西就麻烦。

比如你想做这些实验：

- 把 ReLU 换成 GELU。
- 把普通梯度下降换成 Momentum。
- 加一层 Dropout。
- 把模型从圆形分类换成 MNIST。

如果每次都要钻进 `MLPClassifier` 里改一堆代码，后面会越来越痛苦。

所以这一关的目标是：把你刚才写出来的神经网络，拆成几个可以复用的小零件。

## 2. 如何让模型学会这个？

人写复杂系统时，会自然分工。

一层线性层只负责：

```text
输入 x -> 算 xW+b -> 反向时算 dW、db、dx
```

ReLU 只负责：

```text
负数变 0 -> 反向时挡住负数区域的梯度
```

loss 只负责：

```text
比较预测和标签 -> 给出最后一层梯度
```

optimizer 只负责：

```text
拿到参数和梯度 -> 更新参数
```

这就是深度学习框架的雏形。PyTorch 当然复杂得多，但味道是一样的。

## 3. 需要引入哪些数学对象

### Linear

$$
Y=XW+b
$$

反向传播时：

$$
dW=X^T dY
$$

$$
db=\mathrm{mean}(dY)
$$

$$
dX=dY W^T
$$

所以 `Linear` 在 forward 时必须记住输入 `X`，否则 backward 算不出 `dW`。

### ReLU

$$
\mathrm{ReLU}(x)=\max(0,x)
$$

反向传播：

$$
dX=dY\odot 1[x>0]
$$

所以 ReLU 也要保存一个 mask。

### CrossEntropyLoss

它要先做 Softmax，再算：

$$
L=-\sum_k y_k\log(p_k)
$$

反向传播时输出：

$$
dlogits=\frac{p-y}{m}
$$

### Optimizer

SGD：

$$
\theta \leftarrow \theta-\eta g
$$

Momentum：

$$
v_t=\beta v_{t-1}+(1-\beta)g_t
$$

$$
\theta \leftarrow \theta-\eta v_t
$$

注意 Momentum 多了一个状态 `v`。这就是为什么 optimizer 应该是一个对象，而不只是一个函数。

## 4. 对应代码该怎么组织

本任务有两个核心文件：

- `my_dl_lib.py`：放你的小库。
- `mission_2.py`：用小库重新完成圆形分类。

你可以先实现这些：

```text
Linear
ReLU
GELU
CrossEntropyLoss
Sequential
SGD
Momentum
```

为什么要有 `Sequential`？因为我们希望这样写模型：

```python
model = Sequential(
    Linear(2, 16),
    ReLU(),
    Linear(16, 16),
    ReLU(),
    Linear(16, 2),
)
```

这比手动一层层调用清爽很多。

运行：

```bash
cd exercises/block_01_basics/task_02_mini_dl_lib
python mission_2.py
```

## 5. 逐步实现顺序

先写 `Linear`。它有参数，所以要提供：

```python
parameters()
```

返回参数和梯度：

```python
[(self.W, self.dW), (self.b, self.db)]
```

再写 `ReLU`。它没有参数，所以不需要 `parameters()`。

接着写 `CrossEntropyLoss`。它的 `forward` 返回 loss，`backward` 返回最后一层梯度。

然后写 `Sequential`。forward 从前往后，backward 从后往前。

最后写 optimizer。SGD 不需要状态，Momentum 需要为每个参数保存一个 velocity。

## 6. 常见错误和调试方法

参数没有更新：检查 `parameters()` 返回的是不是参数本体。如果你返回的是 copy，optimizer 改了也没用。

loss 不下降：先用 ReLU + SGD 跑通，不要一上来就用 GELU + Momentum + 深网络。

Momentum 没效果：检查 velocity 是不是每次 `step()` 都被重新创建了。它必须保存在 optimizer 里。

最后一层加了 ReLU：分类模型最后通常输出 logits，不要把它截断成非负数。

GELU backward 写错：可以先只实现 forward，把 GELU 作为选做；核心库先跑通更重要。

## 7. 必做实验

用同一个圆形分类数据做对照：

| 实验 | 设置 | 观察 |
| --- | --- | --- |
| baseline | ReLU + SGD | |
| 动量 | ReLU + Momentum | |
| 激活函数 | GELU + Momentum | |
| 宽一点 | 隐藏层 16 改成 32 | |

每次记录 loss 和准确率。再写一句：你觉得这个变化为什么会产生这种结果？

这就是研究者习惯的开头：不是只跑一个模型，而是设计对照。

## 8. 完成后应该能回答的问题

- 为什么 `Linear.forward` 要保存输入？
- 为什么 ReLU 需要 mask？
- loss 的 backward 为什么不需要外部传入 dout？
- optimizer 为什么应该独立于模型？
- Momentum 比 SGD 多保存了什么？
- `Sequential.backward` 为什么要倒序执行？
- 这个小库和 PyTorch 的 `nn.Module` 有什么相似之处？

## 9. 下一任务为什么自然出现

现在你有了一个小库，它能训练二维分类任务。

但我们不能永远玩二维点。下一步把它放到真实图片上试试：MNIST 手写数字。你会发现 MLP 可以做图片分类，但方法很粗暴：它把图片拉平成一长串数字。

这个粗暴感，会把我们带到卷积。

