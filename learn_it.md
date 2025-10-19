# 前言
这个教程的目的是带你入门或者巩固深度学习的基础知识和算法.你应该已经上过或正在上人工智能入门课并且把作业文档丢给 AI 让它生成过一份"完美的作业"了, 现在该到静下心来再学习一遍的时候了. 

本教程将由 markdown 格式呈现. 和李沐的《动手学习深度学习》不同, 我不太喜欢 jupyter 格式, 打开 jupyter 服务器本身便是对学习热情的一种阻碍, VS Code 的 jupyter 插件又老是出 bug. 因此,这里不会塞满完整代码,而是以公式、思路与关键片段为主,帮助你建立直觉和理解.如果你实在敲不出来,你可以参考文件夹里的reference文件夹里的代码,这是我初学深度学习时实现的代码,这篇教程就是参考我入门时的代码写成的,在细节处会有些不同.

本教程的难度设定是基于获得AI深度辅助的考量而制定的.我们鼓励善使用AI工具来完成代码,从而达到快速学习的目的.

同时, 部分复杂内容会贴出相关链接以供学习, 我相信这些复杂内容别人会讲的比我清楚许多. 如果有建议请尽快告知我.

本教程会频繁更新,每次使用前请更新版本:
```bash
git pull origin main
```
这不会影响到你已经完成的代码文件.

那么让我们开始吧

---
---

# 内容简介
本教程将会分为三个板块循序渐进
## y = ax + b! 神经网络到底是什么?
#### 此板块你将从零开始发明一个神经网络, 只能用 numpy 实现, 包含: 前向传播, 反向传播, 梯度计算, 优化器, 激活函数, 归一化...
## 这是飞机还是轮船? 用 ResNet 分类物体!
#### 你或许早就了解过 ResNet 了, 但这次需要你使用之前你搓出来的深度学习库实现 ResNet. 你需要再手搓出那些常用的深度学习模块, 比如 BatchNorm 归一化, Adam 优化器. 你在调参的时候可能见过这些, 这次轮到你实现了
## apple is __ ? __ 注意力是你所需要的
#### 最后, 你需要使用 PyTorch 库(当然不能用 nn.attention)实现 Transformer 模型和 Mamba 模型, 在熟悉 PyTorch API 的同时完成文本生成任务和文本情感分析任务.

---
---

## 我需要准备什么?
- 非 Windows 环境(如果你的设备是 Windows 推荐使用 WSL, macOS 可以直接用, M1 之后的 MacBook 应该能跑得动本教程需要的内容). [WSL 安装](https://learn.microsoft.com/zh-cn/windows/wsl/install)
- 一个你喜欢的编辑器和终端主题
- python基本知识
- 知识搜索能力(比如 numpy 和 PyTorch 的使用方法, 本教程不会教这些, 请询问 AI 或者搜网上教程)

---
---
---

# y = ax + b! 神经网络到底是什么?
什么是神经网络? 神经网络其实就是一个非常复杂的函数, 我们把它叫做 $f$ 吧. 我们现在有一个输入 $x$ ($x$ 和 $y$ 不是单变量, 通常是一个 tensor). [什么是 tensor(张量)?](https://www.bilibili.com/video/BV19C411G7Zw/?share_source=copy_web), 把 x 塞进神经网络, 我们就能得到输出 $y=f(x)$ 了. 这里 y 和 x 可以是几乎任何有关联的事物, 比如图像识别任务, x 是一张图片, y 是图片的内容; chatgpt中, x 是你的prompt和question, y 是chatgpt的回答. 那神经网络是怎么完成这些复杂的任务的?

 一上来可能难以理解, 让我们从简单的任务开始看起来吧.

## 任务一: 拟合一条直线
![alt text](picture/mission_0.png)
**这是一个坐标系, 上面有一些点, 你要让你的电脑告诉你当 $x=5$ 时, $y$ 是多少.**
---

### 一. 如何让电脑看见线?

看着很简单, 你一眼就能看出规律是 $f(x)=x$. 但电脑要怎么知道呢? 你想到了, 既然这些点分布在一条线上, 那描述这条线的函数一定长这样:
$$
 y = ax + b
$$
这便是你的模型. 只要能找到最合适的 $a$ 和 $b$, 问题就解决了.

### 二. 如何衡量最合适?

一个很自然的想法是: 把所有可能的 $a$ 和 $b$ 都试一遍, 看看哪条线离已知的点最近.
但"远近"该如何衡量呢? 于是你想到了均方误差(Mean Squared Error, MSE).

它的计算方式是这样的:
1. 对于每一个已知点 $(x_i, y_i)$, 计算出模型的预测值 $\hat{y}_i = ax_i + b$.
2. 计算真实值和预测值的差距 $e_i = y_i - \hat{y}_i$.
3. 为了避免正负误差抵消并放大差距, 我们取误差的平方 $e_i^2$.
4. 最后, 将所有点的误差平方求和再取平均, 就得到了 MSE:
$$
L(a, b) = \frac{1}{n} \sum_{i=1}^{n} (y_i - (ax_i + b))^2
$$

> 恭喜你发明了损失函数(Loss Function)!
> 在这个任务中, Loss 就是 MSE. 它像一把尺子, 衡量了你的模型预测得有多烂. 你的目标就是最小化这个 Loss, 让你的预测尽可能接近真实值.

### 三. 如何高效地找到最优参数?

遍历所有 $a$ 和 $b$ 显然不现实. 你又想, 如果随机猜了一组 $(a,b)$, 计算出 Loss 后, 能不能有方向地去调整 $a$ 和 $b$ 呢?

- 如果增大 $a$ 会让 Loss 变大, 那就应该减小 $a$.
- 如果增大 $b$ 会让 Loss 变小, 那就应该继续增大 $b$.

这听起来就像是在一个山谷里, 想以最快的速度走到谷底. 你只需要环顾四周, 找到最陡的下坡方向, 然后迈出一步.

那一次增加或调整多少呢? 0.001? 0.01? 0.1? 你决定试试 0.01 吧, 但是直接这样加减不是很容易走过头吗? 于是你决定, 每次调整的幅度应该和当前位置的坡度成正比, 这样就能更快更准地接近最优解.

这个最陡的下坡方向不就是你高数课上学的梯度吗?

- 梯度(在这里是偏导数)的绝对值大 → Loss 对这个参数很敏感 → 应该调整得多一些.
- 梯度(在这里是偏导数)的绝对值小 → Loss 对这个参数不敏感 → 应该调整得少一些.

于是, 你计算出 Loss 分别对 $a$ 和 $b$ 的偏导数 $\frac{\partial L}{\partial a}$ 和 $\frac{\partial L}{\partial b}$, 然后沿着梯度的反方向去更新参数:
$$
 a \leftarrow a - \eta \, \frac{\partial L}{\partial a}, \quad
 b \leftarrow b - \eta \, \frac{\partial L}{\partial b}
$$

> 恭喜你发明了梯度下降和学习率!
> - 梯度下降(Gradient Descent): 一种优化算法, 它通过计算 Loss 的梯度来指导参数的更新方向.
> - 学习率(Learning Rate, 记为 $\eta$): 决定了每次参数更新的步长, 它控制了模型学习的速度.

### 四. 动手实现

现在, 你已经拥有了完成任务所需的所有理论! 整理一下, 训练一个模型的完整步骤如下:

1. 初始化参数: 随机选择 $a$ 和 $b$ 的初始值.
2. 前向传播: 使用当前的 $a$ 和 $b$ 计算所有点的预测值 $\hat{y}$.
3. 计算损失: 计算预测值 $\hat{y}$ 和真实值 $y$ 之间的均方误差 MSE.
4. 计算梯度: 计算损失函数关于参数 $a$ 和 $b$ 的偏导数.
5. 更新参数(反向传播): 使用梯度下降算法更新参数 $a$ 和 $b$ 的值.
6. 重复: 重复步骤 2 到 5, 直到达到预定的迭代次数(epochs)或损失不再显著下降.
7. 预测: 使用训练好的模型进行预测, 例如计算 $x=5$ 时的 $y$ 值.

深度学习的本质就是用梯度下降法找到最合适的参数组合, 让模型在训练数据上表现最好,这个过程就是模型的训练. 你现在已经掌握了这个过程的核心概念.

相信在强大的 AI 的帮助下你能很好地完成这个任务. 这就是一个简单的神经网络, 请完成你的代码并保存为 `mission_0.py` 文件.

>这里的 a 和 b 就是神经网络的参数 (parameters)——权重 (weight) 与偏置 (bias) 的最简单形态. 更复杂的神经网络不再是单个 a、b, 而是一组组权重矩阵 W 和偏置向量 b; 训练本质上就是用数据自动找到这些参数的取值, 让损失最小. 现代大模型拥有数十亿 (B) 到上万亿 (T) 个参数, 本质与这里的 a、b 相同, 只是规模更大.

## 任务二: 使用神经网络预测点在圆形内还是圆形外
在第一个任务中, 你让电脑学会了画一条最合适的线. 但是现在的任务可没那么容易了.

我们要做的是:

给定一个平面上的点 $(x,y)$, 让神经网络告诉我们这个点是在圆内还是圆外.

想象一下, 你在纸上画了一个圆, 然后在上面撒了很多点. 点如果在圆里面就标 1, 在外面就标 0. 你的任务就是让电脑自动判断点的归属.

![alt text](picture/data_circle.png)

看起来像是图形分类, 对吧? 那我们先看看能不能用上次的思路来解决.

### 一. 为什么线性模型不行了?

在任务一中, 我们用了一个线性函数 $y = ax + b$, 它能画出一条直线.

那现在我们能不能用一个类似的函数, 比如 $f(x,y) = ax + by + c$ 来判断点在不在圆内呢? 如果 $f(x,y) > 0$, 我们就说点在圆内; 否则在外.

看起来挺合理, 但你很快发现问题: 圆的边界是个曲线, 而上面这个函数的边界是直线. 这意味着无论你怎么调 $a,b,c$, 它都画不出圆, 只能画出一条线.

### 二. 你决定加一层非线性层

那怎么让模型变得弯起来? 你想到——如果我们能让输入先经过一个扭一扭的函数, 再加上线性, 就可能出现曲线边界.

这就是激活函数(activation function).

常见的激活函数有:

- ReLU(Rectified Linear Unit):

  $$
  \mathrm{ReLU}(x) = \max(0, x)
  $$

  它会把小于 0 的数都变成 0, 让模型产生折线的效果.
  ![alt text](picture/relu.png)
- Sigmoid:

  $$
  \sigma(x) = \frac{1}{1 + e^{-x}}
  $$

  它把输入压缩到 0~1 之间, 通常用于分类任务.
  ![alt text](picture/sigmoid.png)
- Softmax:
 $$
   \mathrm{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}
 $$
 它把一组输入变成概率分布, 通常用于多分类任务.

你可以理解为:
- ReLU 让模型能学出弯曲的边界.
- Sigmoid 让输出变成是/否的概率.
- Softmax 让输出变成多类别的概率分布.

让我们把一个线性函数和 ReLU 结合起来, 看看会发生什么:
这是一个有一层隐藏层、神经元数量为 5 的简单神经网络结构:
![alt text](picture/relu_network_structure.png)
![alt text](picture/relu_approximation.png)
这条曲线看起来就不像直线了, 而是有折角的曲线.

那如果隐藏层的神经元足够多, 并且用多个隐藏层去堆叠使得曲线更加复杂, 是不是就能画出接近圆形的边界了?

### 三. 交叉熵损失与分类输出
在这个二分类问题里, 模型需要输出在圆内的概率. 线性层给出的是任意实数 logits($-\infty, +\infty$), 但概率必须在 $[0,1]$. 所以我们需要:
- 一个把 logits 变成概率的门(激活函数).
- 一个能衡量概率预测好坏的**损失函数**.

为何不用 MSE?
- MSE 更适合回归; 对概率的差异不够敏感, 无法体现“0.9 比 0.6 更接近标签 1”的语义.

定义交叉熵损失(概率友好)
- 二分类交叉熵(BCE, 单样本): 
$$
L = -\big[y\log(\hat{y}) + (1-y)\log(1-\hat{y})\big]
$$ 
其中 $y\in\{0,1\}, \, \hat{y}$ 是在圆内的概率.你可以把不同的预测值和真实值带入到公式中, 体会 BCE 如何衡量预测的好坏.
比如:

| 真实值 y | 预测值 $\hat{y}$ | BCE 损失 L |
| -------- | ---------------- | ---------- |
| 1        | 0.9              | 0.1054     |
| 1        | 0.6              | 0.5108     |
| 0        | 0.4              | 0.5108     |
| 0        | 0.1              | 0.1054     |

现在直接用我们任务中的例子让你直观了解loss如何计算:

你的最后一层全连接层输出了一个logit值z,让我们假定它是2.0吧!

我们想知道这个点在圆内的概率是多少,于是我们用sigmoid函数把z变成概率:

$$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}} = \frac{1}{1 + e^{-2.0}} \approx 0.8808$$

假设这个点实际上是在圆内的,所以真实标签y是1.

现在我们可以计算BCE损失:

$$L = -[y\log(\hat{y}) + (1-y)\log(1-\hat{y})] = -[1\log(0.8808) + (1-1)\log(1-0.8808)] $$
$$ = -\log(0.8808) \approx 0.1269$$

看着很复杂,仔细一看会发现这是很符合直觉的

这个损失值表示模型在这个样本上的预测与真实标签之间的差距. 损失值越小, 说明模型越自信地做出了正确的预测.

- 多分类交叉熵(CE): 
  
$$
L = -\sum_k y_k\log(p_k)
$$ 

其中 $\mathbf{p} = \mathrm{softmax}(\mathbf{z})$, $\mathbf{y}$ 为 one-hot.
[什么是one-hot(独热编码?)](https://zhuanlan.zhihu.com/p/634296763)

虽然这里不用,但还是举个例子让你直观理解一下,毕竟大部分任务都是多分类任务

假设我们有一个三分类问题,模型的最后一层输出了三个logit值:
$$\mathbf{z} = [2.0, 1.0, 0.1]$$

我们想把这些logit值变成概率分布,于是我们使用softmax函数:

$$p_k = \frac{e^{z_k}}{\sum_j e^{z_j}}$$

计算得到:

$$p_0 = \frac{e^{2.0}}{e^{2.0} + e^{1.0} + e^{0.1}} \approx 0.6590$$

$$p_1 = \frac{e^{1.0}}{e^{2.0} + e^{1.0} + e^{0.1}} \approx 0.2424$$

$$p_2 = \frac{e^{0.1}}{e^{2.0} + e^{1.0} + e^{0.1}} \approx 0.0986$$

所以概率分布为:

$$\mathbf{p} \approx [0.6590, 0.2424, 0.0986]$$

假设真实标签是类别1,对应的one-hot编码为:

$$\mathbf{y} = [0, 1, 0]$$

现在我们可以计算CE损失:

$$L = -\sum_k y_k\log(p_k) = -[0\log(0.6590) + 1\log(0.2424) + 0\log(0.0986)] $$

$$ = -\log(0.2424) \approx 1.4170$$

这个损失值表示模型在这个样本上的预测与真实标签之间的差距. 损失值越小, 说明模型越自信地做出了正确的预测.

在二分类任务中也可以使用多分类交叉熵, 只不过需要把输出变成两维:
- 输出层: 2 维 logits.
- 概率: $\mathbf{p} = \mathrm{softmax}([z_0, z_1])$.
- 标签: one-hot 编码, 如 [1,0] 或 [0,1].

从 logits 变成概率: 两条常见路径
1. Sigmoid: $\sigma(z) = 1/(1+e^{-z})$, 用于单输出的二分类.
2. Softmax: $p_k = e^{z_k}/\sum_j e^{z_j}$, 用于多输出的多分类(包含二分类的 2 类特例).

二分类的两种等价实现(你选其一即可)
1. 方案 A: 单输出 + Sigmoid + BCE($\hat{y}=\sigma(z)$).
2. 方案 B: 双输出 + Softmax + CE($\mathbf{p}=\mathrm{softmax}([z_0, z_1])$).
它们在二分类上是等价建模, 只是表达不同. 后续任务约定使用方案 B(2 维 logits + Softmax + CE), 便于扩展到多类.

数值稳定与现代实践
1. 计算 softmax 前先减去每行最大值(稳定).
2. 现代框架常用带 logits 的损失:
   - 二分类: 线性输出 + BCEWithLogits(内部自带 sigmoid 与稳定项).
   - 多分类: 线性输出 + CrossEntropy(内部自带 softmax 与稳定项).

本任务的明确选择(与后文保持一致)
- 输出层: 2 维 logits → softmax 概率.
- 损失: 交叉熵(CE, 对 batch 取均值).
- 隐藏层: ReLU.
- 标签: 需要转换为 one-hot([1,0] 或 [0,1]).

### 四. 实现多层感知机 (MLP)

既然一条线画不出圆, 那如果让多条“线”先各自处理输入, 再把结果混在一起, 或许就能画出弯曲的边界. 这就得到一个最简单的两层网络:

- 输入层: 输入两个值 (x, y).
- 隐藏层: 有 h 个神经元, 第 i 个神经元计算
  $$ z_i = \mathrm{ReLU}(a_i x + b_i y + c_i). $$
- 输出层: 将这些隐藏单元的输出线性组合得到 2 维 logits, 再用 softmax 变成概率
  $$ \hat{\mathbf{y}} = \mathrm{softmax}(\mathbf{w}_1 z_1 + \cdots + \mathbf{w}_h z_h + \mathbf{b}). $$

不过手写每个神经元既笨又容易出错, 我们更推荐矩阵化的表达.

矩阵化写法(单隐藏层)
- 设一个 batch 的输入为 $\mathbf{X} \in \mathbb{R}^{m\times 2}$, 隐藏层宽度为 $h$.
- 参数: $\mathbf{W}_1 \in \mathbb{R}^{2\times h},\; \mathbf{b}_1 \in \mathbb{R}^{1\times h}$; 输出层 $\mathbf{W}_2 \in \mathbb{R}^{h\times 2},\; \mathbf{b}_2 \in \mathbb{R}^{1\times 2}$.
- 前向:
  $$\mathbf{Z}_1 = \mathbf{X}\,\mathbf{W}_1 + \mathbf{b}_1\;(m\times h), \quad \mathbf{H}_1 = \mathrm{ReLU}(\mathbf{Z}_1).$$
  $$\mathrm{logits} = \mathbf{H}_1\,\mathbf{W}_2 + \mathbf{b}_2\;(m\times 2), \quad \hat{\mathbf{y}} = \mathrm{softmax}(\mathrm{logits}).$$

如果一层不够, 我们可以再叠一层, 例如我们把结构定为 2-4-4-2:
- 输入: X = $(x,y)$.
- 第 1 层: $\mathbf{Z}_1 = \mathbf{X}\,\mathbf{W}_1 + \mathbf{b}_1,\; \mathbf{H}_1 = \mathrm{ReLU}(\mathbf{Z}_1)$.
- 第 2 层: $\mathbf{Z}_2 = \mathbf{H}_1\,\mathbf{W}_2 + \mathbf{b}_2,\; \mathbf{H}_2 = \mathrm{ReLU}(\mathbf{Z}_2)$.
- 输出层: $\mathrm{logits} = \mathbf{H}_2\,\mathbf{W}_3 + \mathbf{b}_3,\; \hat{\mathbf{y}} = \mathrm{softmax}(\mathrm{logits})$.

#### 权重矩阵到底是什么?
- 把任务一中的 $y = ax + b$ 推广到多维输入: 当输入是向量/矩阵时, 标量 a 就要“长成”一个矩阵, 这样才能一次性对所有维度做线性组合. 这就是权重矩阵的由来.
- 形状规则: $\mathbf{W}$ 的形状始终是 in_dim × out_dim. 矩阵乘法会把 $(m\times \text{in})$ 变成 $(m\times \text{out})$.
- 列就是“一个神经元”的权重: $\mathbf{W}_1 \in \mathbb{R}^{2\times h}$ 的第 i 列 $[w_{1i}, w_{2i}]^\top$ 对应隐藏层第 i 个神经元对输入两个特征的线性加权.

以 2→4 为例, 一条样本 $\mathbf{x}=[x, y]$ 的第一层计算是
$$
\mathbf{W}_1 = \begin{bmatrix}
 w_{11} & w_{12} & w_{13} & w_{14} \\
 w_{21} & w_{22} & w_{23} & w_{24}
\end{bmatrix},\quad
\mathbf{b}_1 = [b_1, b_2, b_3, b_4],\quad
\mathbf{z} = \mathbf{x}\,\mathbf{W}_1 + \mathbf{b}_1 = [z_1, z_2, z_3, z_4].
$$
随后 $\mathbf{h}=\mathrm{ReLU}(\mathbf{z})$ 进入下一层. 多层网络就是把“线性投影 + 非线性”反复堆叠, 例如 2→4→4→2, 逐层拉伸/折叠输入空间, 以刻画更复杂的决策边界.

这就是多层感知机(Multi-Layer Perceptron, MLP).

### 五. 批次(batch),轮数(epoch)和训练集

#### batch和epoch

上次的任务里, 你可能直接用 7 个点训练. 可这次你有几千个点. 要是每次都计算所有点的梯度, 会非常慢.假设你有 10000 个点, 每次计算梯度都要算 10000 次, 太慢了,不知道要到什么时候才会收敛.

那我是不是可以对每个点单独计算梯度, 然后更新参数?
显然也不行. 因为每个点的梯度都不一样, 你更新完一个点的参数, 下一个点的梯度就不对了.这就像你在爬山, 每次只能看见一个点的坡度, 结果你走着走着就迷路了.


于是你决定——那我每次只用一小部分点更新参数不就行了?我把这些点叫做一个 batch(批次). 计算这个 batch 上的平均梯度, 然后更新参数. 这样既能保证每次更新的方向大致正确, 又能加快训练速度.

这就是 batch(批次). 比如你有 1000 个点, 每次拿 100 个来训练, 那就要分 10 个 batch. 这里的 batch_size 就是 100.

训练完这 1000 个点一遍, 叫一个 epoch(轮次). 训练 100 个 epoch, 就是把数据看了 100 遍.

#### 训练集和测试集
当你训练完一个epoch时,通常你会想记录看看模型训得怎么样了.那怎么看?看loss的值吗? 当然可以, 但更直观的是看准确率(accuracy). 比如, 你有 1000 个点, 模型预测对了 850 个, 那准确率就是 85%.

但是你发现一个问题,你要实际解决的问题通常不会和训练集上的数据一致,训练集的准确率并不能完全代表模型的真实能力.这就需要引入测试集来评估模型的泛化能力.

所以你通常需要一个训练集(train set)和一个测试集(test set). 训练集用来更新模型参数, 测试集用来评估模型性能. 你可以把数据随机分成 80% 的训练集和 20% 的测试集,每次训练完一个 epoch 后, 在测试集上计算准确率, 这样就能看到模型在未见过的数据上的表现.




### 六. 训练过程(完整逻辑)

好了, 现在你的脑子里的神经网络完整了. 它的训练流程大致是这样的:

1. 初始化参数: 给每个神经元的权重和偏置随机赋值(建议 He 初始化, 偏置 0).
2. 前向传播(矩阵形式):
   - 第 1 隐藏层:
    $$ \mathbf{z}_1 = \mathbf{W}_1\,[x,y]^\top + \mathbf{b}_1,\quad \mathbf{h}_1=\mathrm{ReLU}(\mathbf{z}_1) $$
   - 第 2 隐藏层:
    $$ \mathbf{z}_2 = \mathbf{W}_2\,\mathbf{h}_1 + \mathbf{b}_2,\quad \mathbf{h}_2=\mathrm{ReLU}(\mathbf{z}_2) $$
   - 输出层 logits:
    $$\mathrm{logits} = \mathbf{W}_3\,\mathbf{h}_2 + \mathbf{b}_3$$
   - 概率: 
    $$\mathbf{p} = \mathrm{softmax}(\mathrm{logits})$$
1. 计算损失(Loss): 使用交叉熵(CE), 对 batch 取均值.
2. 反向传播(Backward): 计算损失对每个参数的梯度.
3. 参数更新: 
   $$ \mathbf{W} \leftarrow \mathbf{W} - \eta\, \mathrm{d}\mathbf{W},\quad \mathbf{b} \leftarrow \mathbf{b} - \eta\, \mathrm{d}\mathbf{b} $$



6. 重复以上步骤: 对每个 batch 都做同样的操作, 一轮结束后再开始下一轮(epoch).
7. 预测: 训练结束后, 输入一个新点, 看看 softmax 输出哪个类别概率更大.

### 七. 结果是什么？

训练好后, 你的模型可以在坐标平面上画出一条漂亮的分界线——它会自动学出一个接近圆形的边界.

你刚刚从线性模型一步步发明出了非线性神经网络(MLP), 并掌握了 batch, epoch, ReLU, Sigmoid, Loss, Gradient Descent 这些核心概念.

接下来请你动手实现:

- 输入: 点的坐标 $(x,y)$.
- 输出: Softmax 分类器输出的 2 维概率(取 argmax 得到类别 0/1).
- 结构: 两层全连接层(含 ReLU 激活), 固定 2-4-4-2.
- 损失函数: 交叉熵(CE).
- 优化: 梯度下降.

数据与可视化要求:
- 用 `utils/createDataAndPlot.py` 生成数据与绘图, 标题注明数据模式(circle/triangle/square).
- 训练/测试 8:2 划分, 报告测试集准确率.
- 训练细节建议:
  - 初始化: 使用简单的正态分布随机初始化,均值为0,标准差为0.1; 偏置初始化为0.
  - 学习率与批量: 如 `lr=0.05~0.2`, `batch_size=64~256`, `epochs=200~2000`, 每轮打乱数据.
  - 监控: 周期性打印训练损失/准确率, 最终报告测试准确率.

把你的代码保存为 `mission_1.py`, 训练后画出分类边界看看效果. 在实践中你会遇到一些细节问题, 参考下面的动手实践引导部分.

---

## 任务二(动手实践引导)

这一节按“先搭网络 → 再讲损失与梯度 → 最后反向传播与实现”来层层递进. 读完你应当清楚每个权重矩阵的含义、形状, 以及每一步张量计算在做什么.

### 一. 先把网络搭起来: 权重矩阵与维度
以固定结构 2-4-4-2 为例(输入 2 维, 隐藏层各 4 维, 输出 2 维). 设一个 batch 大小为 $m$.
- 输入: $\mathbf{X} \in \mathbb{R}^{m\times 2}$
- 第 1 层(全连接 + ReLU)
  - $\mathbf{W}_1 \in \mathbb{R}^{2\times 4}$, $\mathbf{b}_1 \in \mathbb{R}^{1\times 4}$ (按行广播到 $m\times 4$)
  - $\mathbf{z}_1 = \mathbf{X}\,\mathbf{W}_1 + \mathbf{b}_1 \;\Rightarrow\; (m\times 4)$
  - $\mathbf{h}_1 = \mathrm{ReLU}(\mathbf{z}_1) \;\Rightarrow\; (m\times 4)$
- 第 2 层(全连接 + ReLU)
  - $\mathbf{W}_2 \in \mathbb{R}^{4\times 4}$, $\mathbf{b}_2 \in \mathbb{R}^{1\times 4}$
  - $\mathbf{z}_2 = \mathbf{h}_1\,\mathbf{W}_2 + \mathbf{b}_2 \;\Rightarrow\; (m\times 4)$
  - $\mathbf{h}_2 = \mathrm{ReLU}(\mathbf{z}_2) \;\Rightarrow\; (m\times 4)$
- 输出层(全连接 + Softmax)
  - $\mathbf{p} = \mathrm{softmax}(\mathrm{logits}) \;\Rightarrow\; (m\times 2)$ (每行是两个类别的概率, 和为 1)
- 标签: $\mathbf{y}_{\text{onehot}} \in \mathbb{R}^{m\times 2}$ (例如类别 0=[1,0], 类别 1=[0,1]).

为什么这些形状是这样? 直观地把列看作输入维度, 行看作输出维度: $\mathbf{W}$ 的形状就是“输入维度×输出维度”, 矩阵乘法自然把 $m\times \text{in}$ 变成 $m\times \text{out}$.

### 二. 前向传播
- $\mathbf{z}_1 = \mathbf{X}(m\times 2)\cdot\mathbf{W}_1(2\times 4) + \mathbf{b}_1(1\times 4) \Rightarrow m\times 4$
- $\mathbf{h}_1 = \mathrm{ReLU}(\mathbf{z}_1) \Rightarrow m\times 4$
- $\mathbf{z}_2 = \mathbf{h}_1(m\times 4)\cdot\mathbf{W}_2(4\times 4) + \mathbf{b}_2(1\times 4) \Rightarrow m\times 4$
- $\mathbf{h}_2 = \mathrm{ReLU}(\mathbf{z}_2) \Rightarrow m\times 4$
- $\mathrm{logits} = \mathbf{h}_2(m\times 4)\cdot\mathbf{W}_3(4\times 2) + \mathbf{b}_3(1\times 2) \Rightarrow m\times 2$
- $\mathbf{p} = \mathrm{softmax}(\mathrm{logits}) \Rightarrow m\times 2$
- 损失: $L = \mathrm{CE}(\mathbf{p}, \mathbf{y}_{\text{onehot}})$ (对 batch 求均值)

### 三. 损失与激活的搭配: 经典与现代做法

- 经典案例(二分类): Sigmoid + 交叉熵(BCE)
  - 对最后一层线性输出 $z$ 的导数为
  $$
  \frac{\partial L}{\partial z} = \hat{y} - y
  $$ 
  这个值正好是预测值 − 真实值! 这是链式法则下 Sigmoid 与 BCE 导数相互抵消的结果.请自己求导推导一下.
  - 教学意义: 推导简单, 展示激活函数与损失函数的配合.
- 为何如今很少用 Sigmoid 作为隐藏层激活?
  - 梯度消失严重; 输出不以 0 为中心; 收敛慢. ReLU/LeakyReLU 更稳定高效.
- 现代主流:
  - 多分类: 线性输出 + CrossEntropy(内部自动 softmax).
  - 二分类: 线性输出 + BCEWithLogits(内部自动 sigmoid).
  - 隐藏层: ReLU 家族(ReLU, LeakyReLU, GELU 等).
- 还要学的理由:
  - 历史与教学价值; 很多经典论文基于此.
  - 特殊场景仍常见: 多标签分类(每个标签一个 sigmoid+BCE), 注意力权重, 需要压到 [0,1] 的概率值.

### 四. 反向传播的本质: 一条局部导数相乘的流水线

根据导数的链式法则公式
$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x}
$$
我们可以把复杂的导数计算拆成一系列局部导数的乘积. 反向传播就是沿着计算图从输出层往输入层逐层计算梯度的过程. 每一层的梯度计算都遵循:**当前梯度 = 上游梯度 × 本层局部梯度.**

以 2-4-4-2 为例, batch 大小 $m$:
- 最后一层(softmax + CE 给出优雅梯度)
  - $\mathrm{d}\,\mathrm{logits} = (\mathbf{p} - \mathbf{y}_{\text{onehot}}) / m$  (形状 $m\times 2$)
  - $\mathrm{d}\mathbf{W}_3 = \mathbf{h}_2^\top\,\mathrm{d}\,\mathrm{logits}$  (形状 $4\times 2$)
  - $\mathrm{d}\mathbf{b}_3 = \text{按行求和}(\mathrm{d}\,\mathrm{logits})$  (形状 $1\times 2$)
- 传到隐藏层 2
  - $\mathrm{d}\mathbf{h}_2 = \mathrm{d}\,\mathrm{logits}\,\mathbf{W}_3^\top$  (形状 $m\times 4$)
  - $\mathrm{d}\mathbf{h}_2^{\text{pre}} = \mathrm{d}\mathbf{h}_2 \odot 1[\mathbf{z}_2>0]$  (ReLU 掩码, 形状 $m\times 4$)
  - $\mathrm{d}\mathbf{W}_2 = \mathbf{h}_1^\top\,\mathrm{d}\mathbf{h}_2^{\text{pre}}$  (形状 $4\times 4$)
  - $\mathrm{d}\mathbf{b}_2 = \text{按行求和}(\mathrm{d}\mathbf{h}_2^{\text{pre}})$  (形状 $1\times 4$)
- 传到隐藏层 1
  - $\mathrm{d}\mathbf{h}_1 = \mathrm{d}\mathbf{h}_2^{\text{pre}}\,\mathbf{W}_2^\top$  (形状 $m\times 4$)
  - $\mathrm{d}\mathbf{h}_1^{\text{pre}} = \mathrm{d}\mathbf{h}_1 \odot 1[\mathbf{z}_1>0]$  (形状 $m\times 4$)
  - $\mathrm{d}\mathbf{W}_1 = \mathbf{X}^\top\,\mathrm{d}\mathbf{h}_1^{\text{pre}}$  (形状 $2\times 4$)
  - $\mathrm{d}\mathbf{b}_1 = \text{按行求和}(\mathrm{d}\mathbf{h}_1^{\text{pre}})$  (形状 $1\times 4$)
- 更新: $\mathbf{W} \leftarrow \mathbf{W} - \eta\,\mathrm{d}\mathbf{W}$, $\mathbf{b} \leftarrow \mathbf{b} - \eta\,\mathrm{d}\mathbf{b}$.

---


## 任务三 完善你的深度学习库
你现在已经完成前两个任务了, 但是目前模型收敛速度和效果还不够好.你将完善你的深度学习小库, 依次引入: 新的激活函数(SiLU/GeLU) → 优化器(mini-batch SGD、Momentum、Adagrad、RMSProp、Adam) → 归一化(BatchNorm/LayerNorm) → 正则化(Dropout) → 模型初始化.

我们先看你会遇到的典型现象:
- 问题 A: 损失下降很慢, 一段时间后几乎不动; 有时一大片神经元都学不到东西.
- 问题 B: 损失忽上忽下, 对学习率非常敏感, 稳不住.
- 问题 C: 训练集准确率很高, 测试集却很差(过拟合).
- 问题 D: batch 改小就不稳定, 同样的学习率今天能收敛, 明天又发抖.

下面我们先从改动最简单也最直观的激活函数开始

### 一. 激活函数
回忆任务二: 我们用了 ReLU. 它简单高效, 但也有两个常见问题:
- Dead ReLU问题:ReLU神经元在训练时比较容易“死亡”.在训练时,如果参数在一次不恰当的更新后,第一个隐藏层中的某个ReLU 神经元在所有的训练数据上都不能被激活,那么这个神经元自身参数的梯度永远都会是0,在以后的训练过程中永远不能被激活.这种现象称为死亡ReLU问题,并且也有可能会发生在其他隐藏层.
- 不以零为中心：和 Sigmoid 激活函数类似,ReLU 函数的输出不以零为中心,ReLU 函数的输出为 0 或正数,给后一层的神经网络引入偏置偏移,会影响梯度下降的效率.

为了解决这些问题,在relu的基础上发展出了几种改进的激活函数,其中SiLU和GELU在现代深度学习中应用较为广泛:
- **SiLU(Sigmoid Linear Unit)**: 也叫 Swish, 是 ReLU 的平滑版本, 由 Google 提出.SiLU 是由 Sigmoid 函数和线性变换结合而成的激活函数,公式如下：

    $$
    \mathrm{SiLU}(x) = x \cdot \sigma(x)
    $$

    其中,$\sigma(x) = \frac{1}{1 + e^{-x}}$ 是标准的 Sigmoid 函数.简单来说,SiLU 将输入 $x$ 与其通过 Sigmoid 函数的输出相乘.这种设计既保留了非线性特性,又引入了平滑性.

  
- **GELU(Gaussian Error Linear Unit)**: 更“平滑”的门, 在 0 附近软过渡, 现代 Transformer 广泛使用.
  GELU 则基于高斯分布的累积分布函数 (CDF), 其定义为：

  $$
  \mathrm{GELU}(x) = x \cdot \Phi(x)
  $$

  其中,$\Phi(x)$ 是标准正态分布的累积分布函数.由于直接计算 $\Phi(x)$ 较为复杂,实践中通常使用近似公式：

  $$
  \mathrm{GELU}(x) \approx x \cdot \sigma(1.702x)
  $$

    或者更精确的近似：

    $$
    \mathrm{GELU}(x) \approx 0.5x \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)\right)\right)
    $$

GELU 的灵感来源于高斯分布,因此它在处理输入时具有一定的概率特性.


![alt text](picture/silu_gelu.png)
[更多激活函数和数学原理](https://zhuanlan.zhihu.com/p/364620596)

#### 如何选择?
- 实际上你应该一开始就用gelu,transformer和很多现代架构在经过实验证明后都采用了gelu作为默认激活函数.
- 当你实在想不到模型哪里可以改动时, 可以更换激活函数试试

### 二. 优化器 (Optimizer)
在训练神经网络时,我们最先接触的就是最基本的 mini-batch SGD.它的更新规则很简单：
$\theta \leftarrow \theta - \eta g_t$
这里的 $\theta$ 是参数,$\eta$ 是学习率,$g_t$ 是当前批次的平均梯度.
SGD 的思路直接有效,但在地形复杂的损失函数上,比如那种像峡谷一样的曲面,它常常在两边摇摆,很难顺利下降到低谷.
#### 1. 带动量的 SGD（Momentum）
带动量的版本在原始 SGD 的基础上加入了"惯性".可以想象一个小球在山谷中滚动,它不会每次都完全听从当前的梯度方向,而是保留了一部分之前的速度：
$v_t = \beta v_{t-1} + (1-\beta) g_t$
$\theta \leftarrow \theta - \eta v_t$
动量帮助模型沿着平均方向前进,减少震荡,也能加速收敛.参数 $\beta$ 控制惯性的大小,常取 0.9.
#### 2. Adagrad
Adagrad 的想法是：对不同参数使用不同的学习率.如果某个方向上梯度一直很大,算法会记下来,并逐渐减小那一方向的步长；反之,如果梯度很小,就保持较大的步长：
$s_t = s_{t-1} + g_t \odot g_t$
$\theta \leftarrow \theta - \eta \frac{g_t}{\sqrt{s_t + \varepsilon}}$
这种机制让 Adagrad 在稀疏数据上表现不错,比如自然语言处理任务,但它的缺点是学习率会不断变小,后期几乎停止更新.
#### 3. RMSProp
RMSProp 解决了 Adagrad 步长衰减过快的问题.它不再把所有历史梯度都累加,而是对最近的梯度平方做指数加权平均：
$s_t = \rho s_{t-1} + (1-\rho) g_t \odot g_t$
$\theta \leftarrow \theta - \eta \frac{g_t}{\sqrt{s_t + \varepsilon}}$
这样既能自适应不同方向的学习率,又能保持更新的灵活性,常被用于训练循环神经网络.
#### 4. Adam
Adam 是目前最常用的优化器之一,它结合了 Momentum 和 RMSProp 的优点.既记录了梯度的平均趋势,又跟踪了每个方向上梯度大小的变化,还在更新时加上了偏置修正：
$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$
$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t \odot g_t$
$\hat{m_t} = \frac{m_t}{1-\beta_1^t}, \quad \hat{v_t} = \frac{v_t}{1-\beta_2^t}$
$\theta \leftarrow \theta - \eta \frac{\hat{m_t}}{\sqrt{\hat{v_t}} + \varepsilon}$
Adam 在训练初期能快速下降,同时在后期保持较稳定的收敛,是深度学习中的默认选择.
#### 5. AdamW
AdamW 是 Adam 的改进版本.它重新处理了权重衰减(weight decay)的方式.传统 Adam 是把衰减混进梯度更新中,而 AdamW 则直接在参数上施加衰减,更符合正则化的理论意义：
$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$
$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t \odot g_t$
$\hat{m_t} = \frac{m_t}{1-\beta_1^t}, \quad \hat{v_t} = \frac{v_t}{1-\beta_2^t}$
$\theta \leftarrow \theta - \eta \frac{\hat{m_t}}{\sqrt{\hat{v_t}} + \varepsilon} - \eta \lambda \theta$
AdamW 通常在大模型训练中表现更好,比如 Transformer 系列模型,已成为许多框架中的默认配置.

这里的讲述会有点难以理解,你可以观看讲解视频[十分钟搞明白Adam和AdamW,SGD,Momentum,RMSProp,Adam,AdamW](https://www.bilibili.com/video/BV1NZ421s75D/?share_source=copy_web&vd_source=4232eb286ab50446fa5cf1c3eb74b04c)
#### 如何选择优化器

- 如果只是做实验或训练通用模型：Adam 或 AdamW 是安全而高效的选择
- 想严格复现早期论文或控制理论收敛性质：可以尝试 Momentum-SGD
- 面对稀疏特征或特定任务（如NLP的词嵌入）：Adagrad 仍然值得一试

### 三. 归一化（Normalization）

在训练神经网络时, 你可能会发现：同样的学习率、同样的结构,有时训练非常稳,有时却抖得厉害.
一个常见原因是 层输入的分布在不断漂移.
因为每个 batch 的数据不同,每一层看到的输入分布也会跟着变化,这被称为 Internal Covariate Shift (内部协变量偏移).

一个直观例子
- 假设有两个 batch.
  - batch0: [0.1, 0.2, 0.3, 0.4, 0.5] → 均值 0.3, 方差 0.02
  - batch1: [0.5, 0.6, 0.7, 0.8, 0.9] → 均值 0.7, 方差 0.02
- 如果不归一化, 第一层可能这次看到范围是 [0.1, 0.5], 下次变成 [0.5, 0.9]. 模型就像在追着漂移的目标学习, 训练不稳定.
- 归一化把每个 batch 的输入都调整到均值约 0、方差约 1, 各层看到的分布稳定, 训练更平滑.

为什么归一化能抑制梯度消失/爆炸?
- 关键在缩放不变性与梯度的链式传导. 设每层为:
  $$ h_{l+1} = f(W_l\,h_l + b_l) $$
  若 $h_l$ 的尺度过大或过小, 多层叠加后, 梯度在反传中要么指数放大(爆炸), 要么被压扁到 0 附近(消失). 归一化把每层的输出“拉回”到合适的尺度, 有三点直接收益:
  - 梯度传播更稳定: 各层输出方差在 1 附近, 梯度方差也不易指数级放大或缩小.
  - 允许更大学习率: 数值受控, 不易发散, 可更快收敛.
  - 激活保持在有效区间: 例如 Sigmoid 的输入不易饱和在 0 或 1, 梯度更大.

### Batch Normalization (BN)
- 在一个 batch 内, 对每个通道(特征维)做标准化, 再学习可训练的缩放/平移参数 $\gamma,\beta$:
  $$
  \mu_B = \mathrm{mean}(x),\quad \sigma_B^2 = \mathrm{var}(x) \\
  \hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \varepsilon}},\quad y = \gamma\,\hat{x} + \beta
  $$
- 推理阶段使用滑动平均累计的全局 $\mu,\sigma^2$.
- 常见放置: 线性/卷积 → BN → 激活(ReLU/GELU 等).
- 优点: 训练更稳更快; 抑制梯度消失; 可用更大学习率.
- 缺点: 对 batch 大小敏感, batch 太小统计不准.

详细的数学推导观看李沐老师的视频[28 批量归一化【动手学深度学习v2】]( https://www.bilibili.com/video/BV1X44y1r77r/?share_source=copy_web&vd_source=4232eb286ab50446fa5cf1c3eb74b04c)

### Layer Normalization (LN)
- 对每个样本自身在特征维上做标准化, 与 batch 大小无关:
  $$
  \mu = \mathrm{mean}_{\text{feature}}(x),\quad \sigma^2 = \mathrm{var}_{\text{feature}}(x) \\
  \hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \varepsilon}},\quad y = \gamma\,\hat{x} + \beta
  $$
- 适用: 小 batch、RNN、Transformer. 一般放在线性后、激活前(与实现保持一致即可).

#### 如何选择:
- CNN / 大 batch: 优先 BN.
- 小 batch / MLP / Transformer / RNN: 优先 LN.

### 四. 正则化（Regularization）
在训练神经网络时, 你可能会发现：模型在训练集上表现很好, 但在测试集上却很差. 这就是过拟合(Overfitting).

过拟合的原因是模型学到了训练数据中的噪声和细节,而不是数据的整体趋势. 为了防止过拟合, 我们可以使用正则化技术。
### Dropout
Dropout 是一种简单而有效的正则化方法.它的基本思想很简单,在训练过程中, 随机“丢弃”一部分神经元, 使得模型不能过度依赖某些特定的神经元.

举例来说,这是一个权重矩阵:
$$\mathbf{W} = \begin{bmatrix}
 w_{11} & w_{12} & w_{13} & w_{14} \\
 w_{21} & w_{22} & w_{23} & w_{24}
\end{bmatrix}$$

现在把dropout率设为 0.1,那经过随机丢弃后, 可能变成:
$$\mathbf{W'} = \begin{bmatrix}
 w_{11} & 0 & w_{13} & w_{14} \\
 0 & w_{22} & w_{23} & w_{24}
\end{bmatrix}$$

具体来说, 在每次前向传播时, 对每个神经元以概率 $p$ 将其输出设为 0, 即“丢弃”该神经元. 更新权重时, 也只计算未被丢弃的神经元的梯度.这样, 模型在训练时会变得更加鲁棒, 因为它不能依赖于某个特定的神经元.

在测试时, 我们不进行 Dropout, 而是将所有神经元的输出乘以 $(1-p)$, 以保持输出的期望不变.
### L2 正则化
L2 正则化, 也称为权重衰减(Weight Decay), 是另一种常用的正则化方法. 它通过在损失函数中添加一个与权重平方成正比的惩罚项, 来限制模型的复杂度.

具体来说, 如果原始损失函数为 $L$, 则加入 L2 正则化后的损失函数为:
$$
L' = L + \lambda \sum_i w_i^2
$$
举个例子:
假设有一个简单的线性回归模型, 其损失函数为均方误差(MSE):
$$
L = \frac{1}{N} \sum_{j=1}^N (y_j - \hat{y}_j)^2
$$
加入 L2 正则化后, 损失函数变为:
$$
L' = \frac{1}{N} \sum_{j=1}^N (y_j - \hat{y}_j)^2 + \lambda \sum_i w_i^2
$$
其中, $\lambda$ 是正则化强度的超参数, $w_i$ 是模型的权重参数. 通过最小化 $L'$, 模型不仅要拟合训练数据, 还要保持权重较小, 从而防止过拟合.
#### 如何选择:
- Dropout: 适用于大模型和小数据集, 可以显著减少过拟合.
- L2 正则化: 适用于各种模型, 是一种通用的正则化方法.
- 两者结合: 在许多情况下, 同时使用 Dropout 和 L2 正则化可以取得更好的效果.

### 五. 参数初始化（Weight Initialization）
参数初始化是深度学习中的一个重要环节. 合适的初始化方法可以帮助模型更快收敛, 避免梯度消失或爆炸的问题.
#### 1. 为何初始化很重要
- 目标: 让信号与梯度在层间传播时方差不至于指数放大或衰减.
- 对于线性层 z = xW + b, 若 x 的各维独立同分布, 希望 Var[z] 与 Var[x] 同一量级.
- 对 ReLU 一类非线性, 约有 Var[ReLU(z)] ≈ 1/2 Var[z] (因负半轴被截断). 据此推导出合适的权重方差.

#### 2. fan_in / fan_out 定义
- fan_in: 每个神经元(输出维度的一个单元)所连接的输入数目. 对形状为 (in_dim × out_dim) 的全连接权重 W, fan_in = in_dim, fan_out = out_dim.
- 卷积层: fan_in = in_channels × kernel_h × kernel_w, fan_out = out_channels × kernel_h × kernel_w.

#### 3. 常用初始化
- Xavier/Glorot(适合 Sigmoid/Tanh 等“对称”激活)
  - 正态: W ~ N(0, 2/(fan_in+fan_out))
  - 均匀: W ~ U(−√(6/(fan_in+fan_out)), +√(6/(fan_in+fan_out)))
- He/Kaiming(适合 ReLU/LeakyReLU/GELU 等“截断/门控”激活)
  - 正态: W ~ N(0, 2/fan_in)
  - 均匀: W ~ U(−√(6/fan_in), +√(6/fan_in))
- 偏置 b: 通常初始化为 0.

以比较常见的 Kaiming 初始化为例:
我们现在有一个全连接层, 输入维度为 256, 输出维度为 512. 那么 fan_in = 256.
根据 Kaiming 初始化的正态分布公式, 我们可以计算出权重矩阵的标准差:
$$\sigma = \sqrt{\frac{2}{\text{fan\_in}}} = \sqrt{\frac{2}{256}} = \sqrt{\frac{1}{128}} \approx 0.0884$$
因此, 我们可以从均值为 0, 标准差为 0.0884 的正态分布中随机采样, 来初始化权重矩阵 W. 偏置 b 则初始化为 0.

>He和Kaiming初始化的名称都是来自同一个人,他就是何恺明!顺带一提,我们之后要搭的ResNet网络,也是何恺明等人提出的.希望以后在pytorch库中也能看到以你的名字命名的函数.


实践建议
- MLP/CNN + ReLU/GELU: 用 He(Kaiming) 初始化.
- 用 Sigmoid/Tanh 的隐藏层: 用 Xavier 初始化.
- 输出层(logits): 仍可用与隐藏层一致的初始化; 若训练初期不稳定, 可将最后一层权重再整体缩小一到两个数量级(如乘 0.1 或用更小方差), 偏置为 0.
- 归一化层: BN/LN 的缩放 γ ← 1, 平移 β ← 0; 某些残差结构中会将“残差分支”末层 γ 或权重置 0 以便初始为近似恒等映射.

### 六. 实现建议(与你的小库接口对齐)
- 激活函数: 提供前向与反向对输入的梯度. 
- 优化器: 将“计算梯度”与“如何用梯度更新参数”解耦; 为每个参数维护其状态(如 m、v、s), 在每步更新后写回.
- 归一化: BN 需维护 running mean/var(动量 0.9~0.99), 训练/评估两种模式; LN 无 batch 统计, 实现更直观.
- 正则化: Dropout 仅在训练模式生效; 为了可复现, 固定随机种子.

### 七. 动手实现
请完成全部或者部分常用的激活函数、优化器、归一化与正则化方法的实现,自行规划实验测试,记录你想记录的改动.把完整的深度学习库保存在 `my_dl_lib.py` 中,并在 `mission_2.py` 中调用它来完成实验.

做到这里, 你就拥有了一套能训练、能稳定、能泛化的小型深度学习库雏形. 下一步,我们将使用它来搭建经典的resnet网络, 并用它来识别图片中的物体!

---
---
---
# 这是飞机还是轮船? 用 ResNet 分类物体!
在前两个任务中, 你已经掌握了从零开始实现一个简单神经网络的基本方法, 并完善了你的深度学习小库. 现在, 你将使用它来搭建一个经典的 ResNet 网络, 并用它来识别cifar-100数据集中的物体类别

---
---
---
# apple is __ ? __ 注意力是你所需要的