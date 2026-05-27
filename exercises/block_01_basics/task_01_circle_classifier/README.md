# task_01：用 MLP 判断点在圆内还是圆外

## 1. 这个任务解决什么问题

上一关里，一条直线就够用了。现在我们故意换一个一条线解决不了的问题。

给你一个点 `(x, y)`，判断它在圆内还是圆外。

![圆形分类数据](../../../assets/images/data_circle.png)

圆内标 1，圆外标 0。看图你会觉得这很简单：中间一团是一类，外面是一类。但电脑还是只会算数。我们先试试任务一的思路还能不能用。

如果写一个线性函数：

$$
f(x,y)=ax+by+c
$$

然后规定 `f(x,y)>0` 是圆内，`f(x,y)<=0` 是圆外，会发生什么？

你很快会发现，`f(x,y)=0` 的边界永远是一条直线。它可以把平面切成两半，但没法画出一个圆。

这就是我们第一次遇到“模型表达能力不够”的问题。不是训练不努力，而是模型本身画不出你想要的形状。

## 2. 如何让模型学会这个？

人判断点在不在圆内，可能会想到距离：

$$
x^2+y^2 \le r^2
$$

但如果我们直接把这个公式写进程序，就不是让模型学习了。我们想让电脑从数据里自己学出类似的边界。

它缺几个能力：

第一，模型不能只画直线。它要有“弯曲”的能力。

第二，模型不能只输出一个随便的数。分类任务里，我们希望它输出“属于每个类别的概率”。

第三，它需要一种适合分类的 loss。MSE 可以用，但不太顺手。分类更常用交叉熵。

第四，参数变多以后，我们不可能手算每个参数该怎么改。需要反向传播把错误一层层传回去。

这些东西合在一起，就是你这次要写的 MLP。

## 3. 需要引入哪些数学对象

### ReLU：让直线折一下

如果一直做线性变换，再多层也没用。线性套线性，最后还是线性。

所以我们在线性变换后面加一个非线性函数。最常见的是 ReLU：

$$
\mathrm{ReLU}(x)=\max(0,x)
$$

它会把小于 0 的部分砍成 0，大于 0 的部分保留。

![ReLU](../../../assets/images/relu.png)

这一下“折”很关键。多个 ReLU 神经元叠在一起，就能拼出折线边界。折线够多，就可以逼近圆。

![隐藏层结构](../../../assets/images/relu_network_structure.png)

![ReLU 逼近效果](../../../assets/images/relu_approximation.png)

### MLP：把很多小折线组合起来

本任务固定用一个 `2 -> 4 -> 4 -> 2` 的网络。

输入是二维点：

$$
\mathbf{X}\in \mathbb{R}^{m\times 2}
$$

这里 `m` 是 batch 大小。

第一层：

$$
\mathbf{H}_1=\mathrm{ReLU}(\mathbf{X}\mathbf{W}_1+\mathbf{b}_1)
$$

第二层：

$$
\mathbf{H}_2=\mathrm{ReLU}(\mathbf{H}_1\mathbf{W}_2+\mathbf{b}_2)
$$

输出层：

$$
logits=\mathbf{H}_2\mathbf{W}_3+\mathbf{b}_3
$$

注意，logits 不是概率。它只是模型给两个类别打的原始分数。

### Softmax：把分数变成概率

Softmax 把一组分数变成一组概率：

$$
p_k=\frac{e^{z_k}}{\sum_j e^{z_j}}
$$

如果 logits 是 `[0.2, 2.1]`，第二个类别的概率会更大，模型就更倾向于判成类别 1。

写代码时要注意数值稳定：

```python
exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
probs = exp / np.sum(exp, axis=1, keepdims=True)
```

减最大值不会改变 Softmax 结果，但能防止 `np.exp` 爆掉。

### 交叉熵：惩罚错误的自信

分类任务里，我们关心真实类别的概率高不高。

交叉熵：

$$
L=-\sum_k y_k\log(p_k)
$$

如果标签是 one-hot，比如 `[0, 1]`，那这个公式其实只会留下真实类别那一项：

$$
L=-\log(p_{\text{true}})
$$

真实类别概率是 0.9：

$$
-\log(0.9)\approx 0.105
$$

真实类别概率是 0.1：

$$
-\log(0.1)\approx 2.303
$$

所以，模型越自信地错，损失越大。

### 反向传播：把错误分给每个参数

Softmax + 交叉熵有一个很舒服的结果：

$$
\frac{\partial L}{\partial logits}=p-y
$$

也就是说，最后一层的梯度就是“预测概率 - 真实标签”。

然后继续往前传：

$$
d\mathbf{W}_3=\mathbf{H}_2^T dlogits
$$

$$
d\mathbf{H}_2=dlogits \mathbf{W}_3^T
$$

遇到 ReLU 时，要乘一个掩码：

$$
d\mathbf{Z}_2=d\mathbf{H}_2\odot 1[\mathbf{H}_2>0]
$$

这就是反向传播的味道：从后往前，每一层都问自己“上游传来的错误，经过我这里时应该怎么分给我的输入和参数？”

## 4. 对应代码该怎么组织

这次代码比任务一复杂很多。如果所有变量都丢在一个脚本里，很快会乱。所以我们写一个类：

```python
class MLPClassifier:
    ...
```

如果你还不熟悉 class，先抓住一个点：类可以帮我们把模型相关的状态都绑在一起。

比如模型要保存：

- 训练集 `train_set`
- 验证集 `val_set`
- 参数 `W1, b1, W2, b2, W3, b3`
- 中间结果 `H1, H2, logits`
- 超参数 `Learning_rate, batch_size, epochs`
- 训练过程中的 loss

这些变量如果每个函数都传一遍，会非常麻烦。用 `self` 存起来后，类里的函数就可以共享它们。

本任务有三个代码文件：

- `data_creater.py`：生成训练数据和验证数据。
- `Model.py`：定义 `MLPClassifier`。
- `starter.py`：调用数据生成、模型训练和可视化。

运行：

```bash
cd exercises/block_01_basics/task_01_circle_classifier
python starter.py
```

## 5. 逐步实现顺序

### 第一步：先处理数据

数据是三列：

```text
x, y, label
```

坐标直接取出来：

```python
self.X_full = self.train_set[['x', 'y']].values
```

标签要转成 one-hot。因为我们使用的是 2 维 logits + Softmax + CE：

```python
self.y_full = pd.get_dummies(self.train_set['label']).values
```

类别 0 变成 `[1, 0]`，类别 1 变成 `[0, 1]`。

### 第二步：初始化参数

结构是 `2 -> 4 -> 4 -> 2`：

```python
self.W1 = np.random.randn(2, 4) * np.sqrt(2. / 4)
self.b1 = np.zeros((1, 4))
self.W2 = np.random.randn(4, 4) * np.sqrt(2. / 4)
self.b2 = np.zeros((1, 4))
self.W3 = np.random.randn(4, 2) * np.sqrt(2. / 4)
self.b3 = np.zeros((1, 2))
```

这里用了 He 初始化的思路。ReLU 会把一部分输出变成 0，如果初始化太随便，信号可能很快变大或变小。

### 第三步：写工具函数

先写这些：

```python
softmax()
relu(M)
deriv_relu(M)
CE(M_pred, M_true)
```

`deriv_relu` 返回的是 ReLU 的梯度掩码：

```python
return (M > 0).astype(float)
```

### 第四步：写 forward

```python
self.H1 = self.relu(self.X @ self.W1 + self.b1)
self.H2 = self.relu(self.H1 @ self.W2 + self.b2)
self.logits = self.H2 @ self.W3 + self.b3
```

这里最重要的是 shape：

| 变量 | shape |
| --- | --- |
| `X` | `(batch, 2)` |
| `W1` | `(2, 4)` |
| `H1` | `(batch, 4)` |
| `W2` | `(4, 4)` |
| `H2` | `(batch, 4)` |
| `W3` | `(4, 2)` |
| `logits` | `(batch, 2)` |

### 第五步：写 backward

从最后一层开始：

```python
self.dL_dlogits = self.softmax() - self.y
```

再往前：

```python
self.dL_dH2 = self.dL_dlogits @ self.W3.T * self.deriv_relu(self.H2)
self.dL_dH1 = self.dL_dH2 @ self.W2.T * self.deriv_relu(self.H1)
```

再算每个权重的梯度：

```python
self.dL_dW3 = self.H2.T @ self.dL_dlogits
self.dL_dW2 = self.H1.T @ self.dL_dH2
self.dL_dW1 = self.X.T @ self.dL_dH1
```

偏置按 batch 维求和：

```python
self.dL_db3 = np.sum(self.dL_dlogits, axis=0, keepdims=True)
```

最后更新参数。

### 第六步：写 fit

`fit` 要做的事情是：

1. 每个 epoch 打乱数据。
2. 按 batch 切数据。
3. 对每个 batch 做 forward。
4. 计算 loss。
5. backward 更新参数。
6. 记录平均 loss。

### 第七步：写 predict

验证时不更新参数，只做 forward，然后：

```python
return np.argmax(self.softmax(), axis=1)
```

## 6. 常见错误和调试方法

shape 错：每一层都打印 shape。不要只看报错猜。

loss 是 `nan`：先看 Softmax 是否减了最大值，再看 CE 里是否用了 `np.clip`。

loss 不降：学习率先改小。然后检查 `self.softmax() - self.y`、矩阵转置、偏置求和 axis。

预测全是一类：可能是学习率太大，也可能是模型太窄。先把隐藏层从 4 改成 16 试试。

图像看起来反了：确认你画的是预测标签还是真实标签。最好两张图都画。

`self` 搞不清：先把它理解成“这个模型对象自己的小背包”。训练集、参数、中间结果都装在这个背包里。

## 7. 必做实验

| 实验 | 设置 | 观察 |
| --- | --- | --- |
| baseline | `2 -> 4 -> 4 -> 2`, `lr=0.01` | |
| 更宽 | `2 -> 16 -> 16 -> 2` | |
| 大学习率 | `lr=0.1` | |
| 小 batch | `batch_size=5` | |
| 换形状 | 修改 `condition` 为方形或三角形 | |

每次记录三件事：loss、验证集准确率、分类图。

不要只写“成功了”。写清楚它是怎么成功的：loss 是平滑下降，还是抖着下降？图上的边界是圆，还是只学到一坨大概区域？

## 8. 完成后应该能回答的问题

- 为什么线性模型分不开圆？
- ReLU 为什么能让模型画出折线边界？
- logits 和概率有什么区别？
- 为什么 Softmax 前要减最大值？
- one-hot 标签有什么用？
- `p - y` 为什么会出现在 backward 里？
- batch 和 epoch 分别是什么？
- `self.X_full` 和 `self.X` 为什么要分开？
- 为什么 `predict` 不应该更新参数？

## 9. 下一任务为什么自然出现

到这里，你已经写出了一个能训练的 MLP。

但你也应该感觉到了：代码开始臃肿。激活函数、loss、optimizer、模型参数全部挤在 `MLPClassifier` 里。下一次你想换 GELU 或 Adam，就要在这个类里到处改。

所以接下来我们不再急着换任务，而是先整理工具。你要把这些散落的部件拆成一个小型深度学习库。

