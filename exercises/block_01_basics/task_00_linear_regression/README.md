# 第一节：拟合一条直线

![线性回归示意图](assets/mission_0.png)

`starter.py` 用 `Salary_Data.csv` 复现[章文](../../../chapters/01-基础知识.md)中的梯度下降，并把标准化坐标中学到的参数换回薪资单位。这里顺着真实数据和代码往下看，手算的一次更新仍留在章文中。

`Salary_Data.csv` 记录了工作年限与薪资：

```text
x: YearsExperience
y: Salary
```

散点大致落在一条直线附近。一个直接的模型是

$$\hat y=ax+b$$

它可以拟合数据，也能估计没有直接出现在数据表里的取值，例如 $x=5$ 时的薪资。$a$ 控制斜率，$b$ 控制截距；训练过程会逐步调整这两个参数。

## 损失与梯度

对第 $i$ 条数据，残差是 $e_i=y_i-\hat y_i$。均方误差对所有残差平方后取平均：

$$
L(a,b)=\frac{1}{n}\sum_{i=1}^{n}\left(y_i-(ax_i+b)\right)^2
$$

平方避免正负残差相互抵消，也会让较大的误差承担更高代价。对两个参数求导可得：

$$
\frac{\partial L}{\partial a}
=-\frac{2}{n}\sum_{i=1}^{n}x_i\left(y_i-\hat y_i\right),
$$

$$
\frac{\partial L}{\partial b}
=-\frac{2}{n}\sum_{i=1}^{n}\left(y_i-\hat y_i\right)
$$

梯度的符号给出损失上升的方向。梯度下降向相反方向走一小步：

$$
a\leftarrow a-\eta\frac{\partial L}{\partial a},\qquad
b\leftarrow b-\eta\frac{\partial L}{\partial b}
$$

学习率 $\eta$ 决定步长。当前脚本每次都拿全部数据算梯度，这就是 full-batch（全批量）更新。

<div class="widget-mount" data-widget="linear-fit"></div>

## 这些式子在代码里的位置

`starter.py` 给出了一份最小训练循环，代码按五个环节排开：

```text
读取并标准化数据
初始化 a、b
计算预测与 MSE
计算 grad_a、grad_b
更新参数
```

运行：

```bash
cd exercises/block_01_basics/task_00_linear_regression
python starter.py
```

随机初值会改变最前面的几行日志，但正常情况下 loss 会逐渐降到约 `0.042`，恢复后的参数接近：

```text
a ≈ 9450
b ≈ 25792
```

除了看 loss，还可以核对几个直接的量：

- loss 是有限数，并且总体下降；
- 训练后斜率为正；
- `gradient_a0` 对应斜率 $a$ 的导数，`gradient_a1` 对应截距 $b$ 的导数；
- `a[0] * 5 + a[1]` 给出 $x=5$ 时的预测。

## 代码为什么先改变了数据尺度

工作年限大多是个位数，薪资却在几万左右。如果直接训练，误差和梯度的数值会很大；而 $x$ 的均值和尺度又会影响斜率、截距两个方向的梯度比例。所以 `if_normalize=1` 会分别标准化 $x$ 和 $y$：前者让两个参数更容易共用一个学习率，后者把目标和整体梯度压到更好处理的数值范围。

$$x'=\frac{x-\mu_x}{\sigma_x},\qquad y'=\frac{y-\mu_y}{\sigma_y}$$

模型在新坐标里学到 $y'=a'x'+b'$，脚本最后再把参数换回原始的薪资单位：

$$
a=a'\frac{\sigma_y}{\sigma_x},\qquad
b=b'\sigma_y+\mu_y-a\mu_x
$$

末尾打印的 `a` 已经换回原始单位下的斜率和截距，最后一行 `loss` 却仍是标准化坐标中的 MSE，不能拿它和“差了多少元”直接比较。

脚本中有几个值很适合用来看训练的变化。`learning_rate=0.001` 和 `0.1` 的下降速度不同；关掉 `if_normalize` 后，合适的学习率范围也会改变。换一组点，训练循环本身不用变，最后的斜率和截距会跟着数据走：

```python
x = np.array([-2, -1, 0, 1, 2, 3, 4])
y = np.array([-4.4, -3.2, -2.3, -1.1, 0.5, 1.9, 2.2])
```

此时的 $a$、$b$ 已经是最简单的 weight 和 bias；全连接层只是把标量 $a$ 扩展成权重矩阵 $W$，训练方法仍然来自同一套梯度下降。
