# task_00：拟合一条直线

## 1. 这个任务解决什么问题

我们先从一个几乎不会吓到人的任务开始。

你面前有一些点。它们大概排成一条线。你想让电脑回答一个问题：如果 `x = 5`，那么 `y` 大概是多少？

![任务一示意图](../../../assets/images/mission_0.png)

人看到这张图，会很自然地在脑子里画一条线。但电脑不会“看趋势”。它只能拿数字算。于是我们需要先给它一个可以计算的形式：

$$
\hat{y}=ax+b
$$

这就是你的第一个模型。`a` 控制斜率，`b` 控制上下平移。只要这两个数字合适，这条线就能贴近那些点。

所以任务变成了：电脑怎么自己找到合适的 `a` 和 `b`？

## 2. 如何让模型学会这个？

如果让你手画，你可能会先随便画一条线，再看哪里偏了，然后慢慢调整。电脑也可以这么做，只是它需要更明确的规则。

它缺三样东西：

第一，它需要知道“当前这条线长什么样”。这就是模型：

$$
\hat{y}=ax+b
$$

第二，它需要知道“这条线现在有多差”。这就是损失函数。

第三，它需要知道“要让自己变好，参数应该往哪边改”。这就是梯度下降。

注意这个顺序。不是先背梯度下降，而是先发现：如果没有一个更新方向，电脑只能乱猜。

## 3. 需要引入哪些数学对象

先随便给一组参数：

```python
a = 0.3
b = -1.2
```

对每个点 `(x_i, y_i)`，模型会给一个预测：

$$
\hat{y}_i=ax_i+b
$$

预测和真实值之间的差距是：

$$
e_i=y_i-\hat{y}_i
$$

如果直接把误差加起来，会出现一个麻烦：有的点预测高了，有的点预测低了，正负一抵消，看起来好像没错。但其实线画得并不好。

所以我们把误差平方，再取平均：

$$
L(a,b)=\frac{1}{n}\sum_{i=1}^{n}(y_i-(ax_i+b))^2
$$

这就是 MSE，均方误差。

你可以把它想成一把尺子。loss 越大，说明线越离谱；loss 越小，说明线越贴近数据。

现在问题又变成：怎么让 loss 变小？

暴力办法当然是把所有 `a` 和 `b` 都试一遍。但两个参数还勉强能想，神经网络以后有成千上万个参数，根本没法枚举。

于是我们看梯度。对 MSE 来说：

$$
\frac{\partial L}{\partial a}=-2\cdot \mathrm{mean}((y-\hat{y})x)
$$

$$
\frac{\partial L}{\partial b}=-2\cdot \mathrm{mean}(y-\hat{y})
$$

你暂时不必害怕这个公式。它只是在回答：如果我稍微改一下 `a` 或 `b`，loss 会怎么变？

更新时沿着梯度反方向走：

$$
a \leftarrow a-\eta\frac{\partial L}{\partial a}
$$

$$
b \leftarrow b-\eta\frac{\partial L}{\partial b}
$$

这里的 `η` 是学习率。它像步子大小。步子太小，你走很久；步子太大，你可能直接跨过谷底，在两边来回跳。

## 4. 对应代码该怎么组织

打开 `starter.py`，你会看到它做了几件事：

1. 用 `pandas` 读取 `Salary_Data.csv`。
2. 随机初始化两个参数。
3. 可选地对数据做标准化。
4. 重复很多次训练循环。
5. 打印 loss 和最终参数。

运行方式：

```bash
cd exercises/block_01_basics/task_00_linear_regression
python starter.py
```

如果你还没装依赖：

```bash
conda create -n dl-tutorial python=3.10
conda activate dl-tutorial
pip install numpy pandas seaborn matplotlib
```

你需要先认识几个 NumPy/Pandas 用法：

```python
import numpy as np
import pandas as pd

a = np.random.randn(2)
df = pd.read_csv("Salary_Data.csv")
print(df.head())
print(df.shape)
print(df.columns)
```

不要小看这些打印。很多初学者训练半天不对，最后发现列名写错了。

## 5. 逐步实现顺序

建议你真的按这个顺序做：

第一步，只读数据。

```python
df = pd.read_csv("Salary_Data.csv")
print(df.head())
```

第二步，取出输入和目标。

```python
x = df["YearsExperience"]
y = df["Salary"]
```

第三步，随便猜一组参数。

```python
a = np.random.randn(2)
```

这里 `a[0]` 是斜率，`a[1]` 是截距。写成一个数组只是为了方便。

第四步，算预测和 loss。

```python
loss_vector = y - a[0] * x - a[1]
loss = np.sum(loss_vector ** 2) / len(loss_vector)
```

第五步，算梯度。

```python
gradient_a0 = -2 * np.mean((y - a[0] * x - a[1]) * x)
gradient_a1 = -2 * np.mean(y - a[0] * x - a[1])
```

第六步，更新参数。

```python
a[0] = a[0] - learning_rate * gradient_a0
a[1] = a[1] - learning_rate * gradient_a1
```

第七步，把上面这些放进循环，每 100 次打印一次 loss。

如果 loss 一直下降，恭喜，你已经让电脑开始“学”了。

## 6. 常见错误和调试方法

loss 变成 `nan`：先别慌，大概率是学习率太大。把 `learning_rate` 降到 `0.001` 或 `0.0001`。

loss 不下降：检查更新符号。梯度下降是减去梯度，不是加上梯度。

loss 下降很慢：检查数据尺度。工资可能是几万，工作年限只有几到十几，尺度差太大时，训练会很别扭。可以试试标准化：

```python
x_normalized = (x - np.mean(x)) / np.std(x)
y_normalized = (y - np.mean(y)) / np.std(y)
```

读不到 CSV：确认你是在任务目录下运行，或者把路径写对。

打印出来的参数怪怪的：如果你标准化了数据，参数是在标准化空间里学到的。先别急着还原，先确认训练过程本身是对的。

## 7. 必做实验

做下面几组实验，别只看结果，要写一句你看到的现象：

| 实验 | 设置 | 你观察到了什么 |
| --- | --- | --- |
| baseline | `learning_rate=0.001`, `epochs=2000` | |
| 大学习率 | 学习率乘 10 | |
| 小学习率 | 学习率除以 10 | |
| 不标准化 | 关闭标准化 | |

再自己造一组点：

```python
x = np.array([-2, -1, 0, 1, 2, 3, 4])
y = np.array([-4.4, -3.2, -2.3, -1.1, 0.5, 1.9, 2.2])
```

看看模型还能不能学出来。

## 8. 完成后应该能回答的问题

- `a` 和 `b` 为什么叫参数？
- loss 为什么能衡量模型好坏？
- MSE 为什么要平方？
- 梯度告诉了我们什么？
- 学习率太大和太小分别会发生什么？
- 标准化为什么可能让训练更稳？
- 这段代码里，哪部分是 forward，哪部分是 backward？

## 9. 下一任务为什么自然出现

到这里，你让电脑学会了一条线。

但世界不会总是给你直线。下一任务里，你要判断点在圆内还是圆外。圆的边界是弯的，线性模型会立刻露馅。于是我们不得不问：怎么让模型学出弯曲的边界？

