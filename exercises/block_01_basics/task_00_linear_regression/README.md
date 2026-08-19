# 第一节：拟合一条直线

![线性回归示意图](assets/mission_0.png)

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

学习率 $\eta$ 决定步长。当前脚本每次用完整数据计算一次梯度，因此这里的一次循环也就是一次 full-batch 更新。

<div class="widget-mount" data-widget="linear-fit" data-title="调 a 与 b，看损失和梯度"></div>

## 为什么先做标准化

工作年限只有个位到十位，薪资却是几万。两个量级相差很大时，同一个学习率很难同时照顾斜率和截距。`if_normalize=1` 会把 $x$、$y$ 分别变成均值为 0、标准差为 1 的量：

$$x'=\frac{x-\mu_x}{\sigma_x},\qquad y'=\frac{y-\mu_y}{\sigma_y}$$

模型先在标准化后的坐标里学习 $y'=a'x'+b'$，训练结束后再换回原始单位：

$$
a=a'\frac{\sigma_y}{\sigma_x},\qquad
b=b'\sigma_y+\mu_y-a\mu_x
$$

所以脚本末尾打印的 `a` 已经恢复成“薪资 / 工作年限”的斜率；最后一行 `loss` 仍是标准化空间中的 MSE，两者处在不同的单位下。

## 对照代码

`starter.py` 给出了一份最小训练循环，代码结构对应下面五个环节：

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

运行结果可以从几个方面核对：

- loss 是有限数，并且总体下降；
- 训练后斜率为正；
- `gradient_a0` 对应斜率 $a$ 的导数，`gradient_a1` 对应截距 $b$ 的导数；
- `a[0] * 5 + a[1]` 给出 $x=5$ 时的预测。

几种常见改动会呈现不同现象：

1. `learning_rate=0.001` 和 `0.1` 会呈现不同的收敛速度；
2. `if_normalize=0` 会改变合适的学习率范围；
3. 换用下面这组数据时，同一训练循环会收敛到另一组斜率和截距。

```python
x = np.array([-2, -1, 0, 1, 2, 3, 4])
y = np.array([-4.4, -3.2, -2.3, -1.1, 0.5, 1.9, 2.2])
```

这里的 $a$、$b$ 已经是最简单的 weight 和 bias。全连接层只是把标量 $a$ 扩展成权重矩阵 $W$，训练方法仍然来自同一套梯度下降。
