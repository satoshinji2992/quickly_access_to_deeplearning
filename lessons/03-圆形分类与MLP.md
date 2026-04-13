# 圆形分类与 MLP

## 这一章在解决什么

上一章里，一条直线就够用了。但如果你现在要判断一个点是在圆内还是圆外，事情就不一样了。

因为圆的边界不是直线。

这意味着：**单纯的线性模型不够了。**

## 为什么线性模型分不开圆

如果你还沿用上一次的思路，可能会想写一个这样的分类函数：

$$f(x,y) = ax + by + c$$

然后规定：

- `f(x,y) > 0` 就判成圆内
- `f(x,y) <= 0` 就判成圆外

问题在于，这个函数画出来的边界永远是一条直线。

而圆的边界是曲线，所以无论你怎么调 `a,b,c`，都不可能真的把圆分干净。

![圆形分类数据](../assets/images/data_circle.png)

## 为什么需要激活函数

那怎么办？

一种很自然的想法是：先做线性变换，再让结果经过一个会“折一下”的函数。这样拼起来，就有机会画出更复杂的边界。

这就是激活函数的作用。

最常见的几个是：

- ReLU：把负数截成 0
- Sigmoid：把值压到 0 到 1 之间
- Softmax：把一组值变成概率分布

![ReLU 与 Sigmoid](../assets/images/relu.png)
![Sigmoid](../assets/images/sigmoid.png)

当很多个 ReLU 神经元叠在一起时，模型就不再只能画直线，而可以慢慢逼近更复杂的边界。

![隐藏层结构](../assets/images/relu_network_structure.png)
![ReLU 逼近效果](../assets/images/relu_approximation.png)

## 多层感知机到底在做什么

最简单的 MLP 可以理解成这样一套流程：

1. 输入二维坐标 `(x,y)`
2. 让第一层神经元各自做一次线性变换 + 激活
3. 再把这些结果送给下一层
4. 最后一层输出 logits
5. 再用 softmax 或 sigmoid 变成概率

如果写成矩阵形式，就是：

$$\mathbf{Z}_1 = \mathbf{X}\mathbf{W}_1 + \mathbf{b}_1,\quad \mathbf{H}_1 = \mathrm{ReLU}(\mathbf{Z}_1)$$

$$\mathbf{Z}_2 = \mathbf{H}_1\mathbf{W}_2 + \mathbf{b}_2,\quad \mathbf{H}_2 = \mathrm{ReLU}(\mathbf{Z}_2)$$

$$\mathrm{logits} = \mathbf{H}_2\mathbf{W}_3 + \mathbf{b}_3$$

## 前向和反向到底怎么串起来

你可以把任务二想成两股流程：

- 前向传播：算预测
- 反向传播：算梯度

![MLP 前向与反向传播](../assets/images/mlp_forward_backward.svg)

前向传播负责把输入变成输出，反向传播负责回答一个问题：

**如果 Loss 现在不够小，那每个参数该往哪边调？**

这就是神经网络真正学东西的地方。

## 为什么这里要用交叉熵

在分类问题里，我们更关心的是概率对不对，而不是数值差了多少。

所以这一章会用交叉熵，而不是继续用 MSE。

如果是二分类，常见做法有两种：

1. 单输出 + Sigmoid + BCE
2. 双输出 + Softmax + CE

为了和后面多分类任务更统一，这个教程里更推荐第二种。

## batch 和 epoch 是什么

到了这一步，数据量通常已经不是几个点，而是几百几千个样本了。

所以训练时一般不会每次都用全部数据，而是分批：

- 一个 batch：一次更新参数时用到的一小批数据
- 一个 epoch：把整个训练集完整看一遍

这样训练速度更快，也更容易组织实验。

## 这一章最重要的收获

看完这一章，你至少应该明白：

- 线性模型为什么分不开圆
- 激活函数为什么能给模型带来非线性
- MLP 的每一层在做什么
- 分类任务里为什么要用交叉熵
- 前向传播和反向传播如何配合

## 接下来做什么

建议顺着这条线继续：

1. 看配套实践引导：
   [03-任务二实践引导.md](/Users/shenqi/Documents/code/深度学习入门/lessons/03-任务二实践引导.md)
2. 运行和修改练习代码：
   [mission_1.py](/Users/shenqi/Documents/code/深度学习入门/exercises/task_01_circle_classifier/mission_1.py)

当你把这个最小 MLP 跑顺之后，下一章就不再是“再写一个模型”，而是开始思考怎么把这些能力整理成一套小型深度学习库。
