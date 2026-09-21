# 第二节：判断点在圆内还是圆外

`Model.py` 把 Linear 与 ReLU 接成一个 `2→4→4→2` 的 MLP，用多条线性边界拼出封闭区域。[章文](../../../chapters/01-基础知识.md)已经手算过直线为什么分不开圆，这里直接沿 `Model.py` 的前向、损失和反向顺序看数组怎么流动。

给定平面上的点 $(x,y)$，标签直接由这个规则确定：

$$
\text{label}=\begin{cases}
1,&x^2+y^2\le 1,\\
0,&x^2+y^2>1
\end{cases}
$$

![半径为 1 的圆形分类数据](assets/data_circle.png)

上一节的 $y=ax+b$ 只能画直线。圆形分类需要封闭的边界，所以 `Model.py` 在两层 Linear 后各放了一个 ReLU。

## ReLU 先把一条直线折一下

$$\mathrm{ReLU}(z)=\max(0,z)$$

当 $z<0$ 时输出 0，$z>0$ 时保留原值。第一张图看的是这个截断动作；放到网络里，它会出现在 Linear 与 Linear 之间；多个 ReLU 的结果再被后一层组合，才得到第三张图里的分段线性形状。

![ReLU 函数](assets/relu.png)

下图只画了一个小网络，重点是 ReLU 在层与层之间的位置：

![带 ReLU 的小网络](assets/relu_network_structure.png)

当后一层把多个神经元的输出加在一起，折角就能拼成更复杂的边界：

![分段线性函数近似曲线](assets/relu_approximation.png)

交互演示也只用一个隐藏层。改变隐藏层宽度，可以直接看到可用的折角变多后，边界如何跟着变细。

<div class="widget-mount" data-widget="mlp-boundary"></div>

## `Model.py` 的前向计算

代码一次处理一组点，这组点就是 batch。设它有 $m$ 个样本，输入为 $\mathbf X\in\mathbb R^{m\times2}$，`Model.py` 的三个线性层这样接在一起：

$$
\begin{aligned}
\mathbf Z_1 &= \mathbf X\mathbf W_1+\mathbf b_1,
&\mathbf H_1&=\mathrm{ReLU}(\mathbf Z_1),\\
\mathbf Z_2 &= \mathbf H_1\mathbf W_2+\mathbf b_2,
&\mathbf H_2&=\mathrm{ReLU}(\mathbf Z_2),\\
\mathbf Z_3\;(\text{logits}) &= \mathbf H_2\mathbf W_3+\mathbf b_3
\end{aligned}
$$

参数和中间量的 shape 是：

| 名称 | shape | 含义 |
| --- | --- | --- |
| $\mathbf X$ | $(m,2)$ | 一批点的 $x,y$ 坐标 |
| $\mathbf W_1,\mathbf b_1$ | $(2,4),(1,4)$ | 第一层参数 |
| $\mathbf H_1$ | $(m,4)$ | 第一层激活 |
| $\mathbf W_2,\mathbf b_2$ | $(4,4),(1,4)$ | 第二层参数 |
| $\mathbf H_2$ | $(m,4)$ | 第二层激活 |
| $\mathbf W_3,\mathbf b_3$ | $(4,2),(1,2)$ | 输出层参数 |
| logits | $(m,2)$ | 圆外、圆内两个类别分数 |

本仓库把样本放在行上，所以 Linear 权重写成

$$
(\text{in\_dim},\text{out\_dim})
$$

于是 $(m,\text{in})@(\text{in},\text{out})$ 得到 $(m,\text{out})$，偏置 $(1,\text{out})$ 再沿 batch 维广播。有些框架会以转置后的形式存权重，阅读这份代码时以上表的约定为准。

## 代码里的 softmax 与交叉熵

logit 是未归一化的类别分数，可以为任意实数。softmax 将每行分数转成和为 1 的概率：

$$
p_{ik}=\frac{\exp(z_{ik})}{\sum_j\exp(z_{ij})}
$$

实际计算前先减去该行最大值：

$$
\mathrm{softmax}(\mathbf z)=
\frac{\exp(\mathbf z-\max(\mathbf z))}
{\sum_j\exp(z_j-\max(\mathbf z))}
$$

加减同一个常数不会改变结果，却能避免较大 logit 在 `exp` 中溢出。标签在代码里写成 one-hot：类别 0 为 `[1, 0]`，类别 1 为 `[0, 1]`。

batch 平均交叉熵为：

$$
L=-\frac{1}{m}\sum_{i=1}^{m}\sum_{k=1}^{2}y_{ik}\log p_{ik}
$$

因为 one-hot 每行只有一个 1，单个样本的损失就是正确类别概率的负对数。正确类别概率越接近 1，损失越接近 0。

softmax 与交叉熵合在一起求导后，网络末端的梯度可化简为：

$$
\mathrm d\mathbf Z_3=\frac{\mathbf P-\mathbf Y}{m}
$$

除以 $m$ 是因为前向损失对 batch 取了平均。重复除一次会让梯度多缩小 $m$ 倍，完全不除则会让有效学习率随 batch size 改变。

<details>
<summary>二分类为什么没用 Sigmoid？</summary>

Sigmoid 会把一个分数压到 $(0,1)$，常与单输出的二分类交叉熵搭配。当前代码选择两个 logits 与 softmax，是为了让圆形二分类和后面的多分类共用一套写法。两种表达的对照和数值例子已放在[章文的选读段落](../../../chapters/01-基础知识.md)。

![Sigmoid 函数](assets/sigmoid.png)

</details>

## `backward()` 从最后一层往回走

章文用一个标量例子跟过链式法则。换成矩阵后，每层做的事没变。线性层 $\mathbf Z=\mathbf X\mathbf W+\mathbf b$ 收到 $\mathrm d\mathbf Z$ 时，一边收集本层参数的梯度，一边把梯度传给更靠近输入的一层：

$$
\mathrm d\mathbf W=\mathbf X^\top\mathrm d\mathbf Z,
\qquad
\mathrm d\mathbf b=\sum_{i=1}^{m}\mathrm d\mathbf Z_i,
\qquad
\mathrm d\mathbf X=\mathrm d\mathbf Z\mathbf W^\top
$$

ReLU 只让前向时大于 0 的位置通过梯度：

$$
\mathrm d\mathbf Z=\mathrm d\mathbf H\odot\mathbb{1}[\mathbf Z>0]
$$

从 $\mathrm d\mathbf Z_3=(\mathbf P-\mathbf Y)/m$ 出发，代码倒着经过三个线性层：

| 当前位置 | 传回上一层 | 收集参数梯度 |
| --- | --- | --- |
| 输出层 | $\mathrm d\mathbf H_2=\mathrm d\mathbf Z_3\mathbf W_3^\top$ | $\mathrm d\mathbf W_3=\mathbf H_2^\top\mathrm d\mathbf Z_3$，$\mathrm d\mathbf b_3=\sum_i\mathrm d\mathbf Z_{3,i}$ |
| 第二个 ReLU 与 Linear | $\mathrm d\mathbf Z_2=\mathrm d\mathbf H_2\odot\mathbb{1}[\mathbf Z_2>0]$，再算 $\mathrm d\mathbf H_1=\mathrm d\mathbf Z_2\mathbf W_2^\top$ | $\mathrm d\mathbf W_2=\mathbf H_1^\top\mathrm d\mathbf Z_2$，$\mathrm d\mathbf b_2=\sum_i\mathrm d\mathbf Z_{2,i}$ |
| 第一个 ReLU 与 Linear | $\mathrm d\mathbf Z_1=\mathrm d\mathbf H_1\odot\mathbb{1}[\mathbf Z_1>0]$ | $\mathrm d\mathbf W_1=\mathbf X^\top\mathrm d\mathbf Z_1$，$\mathrm d\mathbf b_1=\sum_i\mathrm d\mathbf Z_{1,i}$ |

三层使用的是同一组矩阵关系。以输出层为例，$\mathbf H_2^\top\mathrm d\mathbf Z_3$ 是 $(4,m)@(m,2)=(4,2)$，结果正好与 $\mathbf W_3$ 同 shape；后两层只是重复这件事。

`Model.py` 为了让第一版模型集中在一个文件里，把“算梯度”和“用梯度改参数”都放进了 `backward()`。这是暂时的省事写法，不是说两个动作本来就是一件事。下一节会把它们拆成 Layer 和 Optimizer。

<div class="widget-mount" data-widget="backprop-steps"></div>

## batch、epoch 与验证集

`batch_size=20` 表示每次用 20 个训练样本的平均梯度更新参数。800 个训练样本完整走一遍是一个 epoch，共有 40 次更新；每个 epoch 开始前，代码都会用固定随机数生成器打乱训练顺序。

日志还会计算验证指标，但验证集不参加 `backward()`，它用来观察模型在另一批未参与更新的样本上能否保持表现。若训练准确率继续提高，验证准确率却开始下降，模型通常是在更贴合训练样本，未必能更好地泛化。

这些 CSV 也不是一次生成后永远沿用。`starter.py` 每次都会按当前 `condition` 生成同一个随机样本池，再按类别分成 800 条训练数据和 200 条验证数据，两边坐标不重叠，类别比例也尽量一致。

训练开始前，程序会重新用半径 1 的条件验算每个标签，并检查集合内部的重复点。如果 CSV 还保留着旧标签，或者训练集和验证集发生交叉，代码会先报错，而不是带着坏数据继续训练。

## 运行与日志

```bash
cd exercises/block_01_basics/task_01_circle_classifier
python starter.py
```

正常开头类似：

```text
data check passed: train=800, val=200, overlap=0, positive_ratio=0.401/0.400
Epoch    1 | train_loss=... train_acc=... | val_loss=... val_acc=...
```

默认参数固定了随机种子，最终验证准确率通常在 `0.95` 以上。第一行先核对数据：train/val 应为 800/200，`overlap=0`；后面的损失再总体下降，期间不应出现 `nan` 或 `inf`。

这组数据中，永远猜数量更多的那一类，也能有大约 60% 的准确率。所以“高于 60%”只表明模型超过了这个最简单的基线，还不能说明边界已经学好。

若 `backward()` 的结果可疑，可以直接比较每个 $\mathrm dW$、$\mathrm db$ 与对应参数的 shape。它们不一样时，沿矩阵乘法的内维往回查，通常比反复调学习率更快找到问题。

仓库测试会单独检查标签、分层和集合泄漏：

```bash
python -m unittest tests.test_block1 -v
```

边界由 `starter.py` 中的向量化条件控制。例如，轴对齐方形可以写成：

```python
condition = "(np.abs(x) <= 0.8) & (np.abs(y) <= 0.8)"
```

下一次运行会用同一批坐标重算标签并重建划分，方便比较“只换目标边界”带来的变化。

## 参考资料

- [Stanford CS231n：二维数据上的 softmax 与两层神经网络](https://cs231n.github.io/neural-networks-case-study/)
- [PyTorch：gradcheck 的有限差分原理](https://docs.pytorch.org/docs/stable/notes/gradcheck.html)
