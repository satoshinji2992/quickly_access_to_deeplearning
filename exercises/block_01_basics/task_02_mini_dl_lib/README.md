# 第三节：整理一个小型深度学习库

圆形分类已经跑通，但 `Model.py` 有个很明显的麻烦：线性计算、ReLU、损失、反向传播和参数更新全挤在一个类里。想多加一层，或者把 SGD 换成 Momentum，几乎每一段都要跟着改。把已经写过的训练流程拆开以后，三层 Linear 不必复制三份反向公式，优化器也不再埋在模型的 `backward()` 里。各部分的关系如下：

![小型深度学习库的组件关系](assets/mini_dl_lib_split.png)

- `Sequential` 保存一组 Layer；
- Layer 的 `forward` 产生激活，`backward` 接收上游梯度，再把梯度传给更靠近输入的一层；
- Loss 比较 logits 与标签，给出反向传播的起点；
- Optimizer 持有参数与梯度的引用，负责更新 Layer 中的参数。

起始文件 `my_dl_lib.py` 留了七类组件：`Linear`、`ReLU`、`GELU`、`CrossEntropyLoss`、`Sequential`、`SGD` 和 `Momentum`。`mission_2.py` 的实际路径只经过 Linear、ReLU、CrossEntropyLoss、Sequential 和 Momentum，SGD 与 GELU 则保留为同类接口的对照实现。

本页前半部分围绕这七类展开。后半部分的 AdamW、归一化和 Dropout 不影响 `mission_2.py` 运行，它们是后续 Block 会用到的选读参考。

## 一层需要记住什么

一个层至少要回答两个问题：

```python
out = layer.forward(x)
dx = layer.backward(dout)
```

`forward` 可以缓存反向所需的量，例如 Linear 的输入、ReLU 的正值 mask。`backward` 不需要知道整个损失函数，只使用上游传来的 `dout` 和自己的缓存。

有参数的层还返回 `(参数, 梯度)` 对：

```python
[(W, dW), (b, db)]
```

优化器在初始化时保存了这些 NumPy 数组的引用。因此 `backward` 采用原地写入，保留现有梯度数组：

```python
self.dW[...] = self.x.T @ dout
self.db[...] = np.sum(dout, axis=0, keepdims=True)
```

如果改成 `self.dW = ...` 并换掉整个数组，优化器可能仍指向旧的 `dW`：日志里梯度非零，参数却不更新。后面的卷积层也会遇到同样的引用问题。

## 先拆出 Linear 与 ReLU

Linear 的前向计算为：

$$
\mathbf Y=\mathbf X\mathbf W+\mathbf b
$$

其中 $\mathbf X:(B,D_{in})$，$\mathbf W:(D_{in},D_{out})$，$\mathbf b:(1,D_{out})$，输出为 $(B,D_{out})$。

收到 $\mathrm d\mathbf Y$ 后：

$$
\mathrm d\mathbf W=\mathbf X^\top\mathrm d\mathbf Y,
\qquad
\mathrm d\mathbf b=\sum_{i=1}^{B}\mathrm d\mathbf Y_i,
\qquad
\mathrm d\mathbf X=\mathrm d\mathbf Y\mathbf W^\top
$$

ReLU 的计算更短：

$$
y=\max(0,x),\qquad
\frac{\partial y}{\partial x}=\mathbb{1}[x>0]
$$

前向保存 `x > 0` 的布尔 mask，反向返回 `dout * mask`。在 $x=0$ 处导数没有唯一值，本项目与常见框架一样取 0。

## CrossEntropyLoss

Loss 接收 logits 和 one-hot targets。先按行减去最大 logit，再计算 softmax：

$$
\mathbf P=\mathrm{softmax}(\mathbf Z-\max(\mathbf Z,\text{axis}=1))
$$

前向返回 batch 平均交叉熵：

$$
L=-\frac1B\sum_{i=1}^{B}\sum_k y_{ik}\log p_{ik}
$$

反向返回：

$$
\frac{\partial L}{\partial\mathbf Z}=\frac{\mathbf P-\mathbf Y}{B}
$$

`CrossEntropyLoss.backward()` 不接收 `dout`，因为损失是计算图末端的标量，它直接使用前向缓存的 `probs` 和 `targets` 产生第一份梯度。

实现中的轴和缩放需要与公式一致：softmax 在类别轴 `axis=1` 上归一化，`max` 和 `sum` 都保留长度为 1 的维度，才能按行广播；`log` 前加很小的 `eps` 可以避免 `log(0)`。损失已经对 batch 取了平均，反向中的 batch size 也只除这一次。

## Sequential 怎样串起各层

前向按保存顺序调用：

```python
for layer in self.layers:
    x = layer.forward(x)
```

反向传播按逆序调用各层：

```python
for layer in reversed(self.layers):
    dout = layer.backward(dout)
```

`parameters()` 已经给出，它只收集实现了 `parameters` 的层，所以 ReLU、GELU 这类无参数层会被自然略过。以 `2 → 16 → 16 → 2` 为例，三层 Linear 每层各贡献 `(W, dW)` 和 `(b, db)`，因此返回列表共有 6 个“参数、梯度”对。

`train()` 和 `eval()` 同样逐层转发：子层实现了同名方法就调用，没有就跳过。当前的 Linear 和 ReLU 在训练、验证时行为一样；加入 Dropout 或 BatchNorm 后，两种模式才会真正分开。

## SGD 与 Momentum

最基本的 SGD 对每个 `(value, grad)` 做原地更新：

$$\theta\leftarrow\theta-\eta g_t$$

Momentum 还为每个参数保存一个同 shape 的速度。当前代码采用下面的约定：

$$
v_t=\beta v_{t-1}+(1-\beta)g_t,
\qquad
\theta\leftarrow\theta-\eta v_t
$$

有些教材写成 $v_t=\beta v_{t-1}+g_t$，与上式的缩放不同；这份实现采用前一种定义，与构造函数中 `beta` 的语义保持一致。SGD 不保存历史状态，Momentum 则让连续多个 batch 中方向一致的梯度逐步积累，并平滑相互抵消的方向，实际收益取决于损失曲面以及学习率、$\beta$ 的组合。

## 把五个组件串起来

`mission_2.py` 串起的路径是 `Linear → ReLU → Linear → ReLU → Linear`，损失用 CrossEntropyLoss，更新用 Momentum。这五种组件接通后，就能先运行圆形分类：

```bash
cd exercises/block_01_basics/task_02_mini_dl_lib
python mission_2.py
```

`mission_2.py` 每轮把 2000 个合成点作为一个 full batch，只打印训练损失与训练准确率。这段脚本看的是数据能否走完 `forward → loss → backward → step`，以及参数是否真的发生变化。

它没有验证集；要观察泛化，仍以[圆形分类页](../task_01_circle_classifier/README.md)那组独立数据为准。

日志每 50 轮打印一次：

```text
epoch=0 loss=... acc=...
epoch=50 loss=... acc=...
...
```

若运行时出现这五种组件的 `NotImplementedError`，可以按报错位置回查还没有接通的接口。全部连通后，loss 应该总体下降，准确率会超过约 65% 的多数类基线。

另一种很具体的异常是：梯度已经非零，`optimizer.step()` 后参数却完全不变。这时先看 `dW[...]` 是不是误写成了 `dW = ...`，后者会让优化器继续指向旧的梯度数组。

## shape 对了，`backward` 也可能是错的

更可靠的检查是把解析梯度与有限差分比较。对某个标量参数 $\theta_j$，先把它分别向左、向右挪一个很小的 $\varepsilon$：

$$
g_j^{\text{numeric}}\approx
\frac{L(\theta_j+\varepsilon)-L(\theta_j-\varepsilon)}{2\varepsilon}
$$

这个数字应当接近 `backward` 写入的梯度。检查时用很小的双精度输入，关闭随机操作，并避开 ReLU 的 0 点。$\varepsilon$ 可以从 `1e-5` 或 `1e-6` 开始：太大时不再像“局部”变化，太小则会被浮点舍入误差淹没。

## ReLU 之外，为什么还留着 GELU

圆形分类用 ReLU 就够了。它的好处是简单，但负半轴的梯度始终是 0；如果某个神经元对所有训练样本都落在这一边，它就可能很难再回到激活区。SiLU 和 GELU 在 0 附近的过渡更平滑：

$$
\mathrm{SiLU}(x)=x\sigma(x)
$$

$$
\mathrm{GELU}(x)\approx\frac{x}{2}\left[1+\tanh\left(\sqrt{\frac{2}{\pi}}
\,(x+0.044715x^3)\right)\right]
$$

![ReLU、SiLU 与 GELU 的形状](assets/silu_gelu.png)

令 $u=\sqrt{2/\pi}(x+0.044715x^3)$、$t=\tanh(u)$，对应的导数是

$$
\frac{\mathrm d\,\mathrm{GELU}}{\mathrm dx}
=\frac12(1+t)+\frac{x}{2}(1-t^2)
\sqrt{\frac{2}{\pi}}(1+3\cdot0.044715x^2)
$$

它比 Linear 的导数长得多，正好适合用上面的有限差分反查。平滑并不代表在所有模型上都更好；当前圆形分类仍使用 ReLU，后面的 Transformer 会再用到同属平滑激活的 SiLU。`GELU` 留在起始库里，`mission_2.py` 的计算路径不会经过它。

## 后面还会用到的组件（选读）

运行 `mission_2.py` 不需要下面这些类。它们放在根目录的 `common/my_dl_lib.py` 中，供后面的卷积网络和序列模型复用；第一次阅读时可以先跳过，等日志里遇到对应现象再回来查。

<details>
<summary>损失在两边来回摆：优化器还能记住什么？</summary>

Momentum 记住近期梯度的平均方向。Adagrad 另外累加每个参数的历史梯度平方，梯度经常很大的方向会自动缩小步长：

$$
s_t=s_{t-1}+g_t^2,
\qquad
\theta\leftarrow\theta-\eta\frac{g_t}{\sqrt{s_t}+\varepsilon}
$$

它的 $s_t$ 只增不减，训练久了可能走得越来越慢。RMSProp 把“从开始到现在全部累加”换成指数移动平均：

$$
s_t=\rho s_{t-1}+(1-\rho)g_t^2
$$

Adam 同时记录梯度方向 $m_t$ 和梯度平方 $v_t$，并在刚开始时修正两个移动平均的偏差：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$

$$
\hat m_t=\frac{m_t}{1-\beta_1^t},
\qquad
\hat v_t=\frac{v_t}{1-\beta_2^t},
\qquad
\theta\leftarrow\theta-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon}
$$

AdamW 再把 weight decay 从自适应梯度的缩放中拆出来。对 Adam 这类优化器，在损失里加 L2 惩罚和直接衰减权重一般不等价。这些优化器没有固定的速度排名，直接拿同一个学习率比较也未必公平。

</details>

<details>
<summary>batch 一换，中间数值就变得很厉害：BN 和 LN 在沿哪个轴统计？</summary>

`BatchNorm1D` 面向 $(B,F)$ 输入，对每个特征收集同一 batch 里的均值和方差：

$$
\hat x=\frac{x-\mu_B}{\sqrt{\sigma_B^2+\varepsilon}},
\qquad
y=\gamma\hat x+\beta
$$

训练时，BatchNorm 还会积累 `running_mean` 和 `running_var`，验证时改用这组运行统计，这就是 `train()` 和 `eval()` 不能永远空着的原因。LayerNorm 不沿 batch 统计，而是对每个样本的最后一个特征维归一化；两者的统计轴不同，不能只看输入 shape 相同就互换。

到 Transformer 里，序列常写成 $(B,T,F)$：$B$ 是 batch size，$T$ 是序列长度，$F$ 是特征数。LayerNorm 的均值和方差沿 $F$ 计算，$\gamma,\beta$ 在所有 $B,T$ 位置共享，所以参数梯度要对前两个轴求和。

</details>

<details>
<summary>训练准确率继续涨，验证准确率却掉了：Dropout 和 L2 各改了什么？</summary>

标准 Dropout 遮蔽的是激活值，不是权重矩阵。设丢弃概率为 $p$、保留概率为 $q=1-p$，inverted dropout 在训练时计算

$$
\mathbf H'=\frac{\mathbf M\odot\mathbf H}{q},
\qquad M_{ij}\sim\mathrm{Bernoulli}(q)
$$

除以 $q$ 使 $\mathbb E[\mathbf H']=\mathbf H$。反向要复用同一个 mask；`eval()` 时直接返回输入，不再随机遮蔽或补做缩放。随机将权重元素置零是 DropConnect，不是 Dropout。

L2 正则化不改激活，而是在原损失后加一项权重平方和：

$$
L'=L+\lambda\sum_i w_i^2
$$

这会让过大的权重额外增加损失，但“权重更小”不等于一定能改善验证结果。它和 Dropout 是两种不同的改动，不需要默认捆在一起。

</details>

<details>
<summary>还没训练，激活就一层层变大或变小：初始权重该有多大？</summary>

`Linear` 根据输入连接数 `fan_in` 设置起始尺度。当激活函数是 ReLU 时，当前代码用 Kaiming/He 正态初始化：

$$
\varepsilon\sim\mathcal N(0,1),\qquad
W=\sqrt{\frac{2}{\text{fan\_in}}}\,\varepsilon,
\qquad b=0
$$

也就是说，权重方差是 $2/\text{fan\_in}$，传给 `np.random.normal` 的标准差则是 $\sqrt{2/\text{fan\_in}}$。这个尺度尽量避免信号在层间立即放大或衰减，但不保证后续训练永远稳定，也不能代替学习率选择。

Kaiming 是何恺明名字的英文写法，下一个 Block 的 ResNet 也来自他和合作者的工作。

如果使用 Tanh 这类正负两侧较对称的激活，常见的 Xavier 正态初始化会采用方差 $2/(\text{fan\_in}+\text{fan\_out})$。初始化需要结合激活函数的行为来选，没有一组方差适合所有网络。

</details>

这些实现最值得对照的仍然是两件事：前向为反向缓存了什么，以及参数梯度沿哪些轴求和。Block 2 遇到卷积与 BatchNorm，Block 3 遇到序列与 LayerNorm 时，会再把它们放回具体模型里。

## 参考资料

- [Stanford CS231n：初始化、正则化与梯度检查](https://cs231n.github.io/neural-networks-2/)
- [PyTorch：`gradcheck` 的数值与解析梯度比较](https://docs.pytorch.org/docs/stable/notes/gradcheck.html)
- [Dive into Deep Learning：Dropout 从零实现](https://d2l.ai/chapter_multilayer-perceptrons/dropout.html)
