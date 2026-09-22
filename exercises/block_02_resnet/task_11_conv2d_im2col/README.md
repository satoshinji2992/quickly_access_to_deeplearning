# 从一个卷积窗口到 im2col

[`conv2d.py`](./conv2d.py) 实现 NCHW 卷积。先把一个输出算对，再看代码怎样把许多窗口合成一次矩阵乘法。

## 先手算中心位置

图中的输入是 `5×5`，卷积核是 `3×3`：

![一个可手算的互相关例子](assets/conv2d_explained.png)

$$
X=
\begin{bmatrix}
1&0&2&1&0\\
0&1&1&0&2\\
2&1&0&1&1\\
0&2&1&0&1\\
1&0&1&2&0
\end{bmatrix}
$$

$$
K=
\begin{bmatrix}
1&0&-1\\
1&0&-1\\
1&0&-1
\end{bmatrix}
$$

当 `stride=1, padding=0` 时，中心输出读取输入的第 2～4 行、第 2～4 列：

$$
\begin{aligned}
y_{1,1}
&=1\cdot1+1\cdot0+0\cdot(-1)\\
&\quad+1\cdot1+0\cdot0+1\cdot(-1)\\
&\quad+2\cdot1+1\cdot0+0\cdot(-1)\\
&=3
\end{aligned}
$$

九个位置都这样计算，完整输出为：

$$
Y=
\begin{bmatrix}
0&0&0\\
0&3&-2\\
1&0&0
\end{bmatrix}
$$

<div class="widget-mount" data-widget="conv-sweep"></div>

深度学习框架通常不把卷积核翻转，所以这里严格说是在做二维互相关。框架和教材仍习惯把这一层叫 Conv2D，后文沿用这个名字。

## 多通道只是在窗口里再多一层求和

数组 shape 约定如下：

```text
X  (N,C_in,H,W)
W  (C_out,C_in,K_h,K_w)
b  (C_out,)
Y  (N,C_out,H_out,W_out)
```

对某个输出位置，先分别计算所有输入通道的窗口贡献，再加起来。比如 RGB 三个通道分别给出 `2、−1、4`，偏置是 `0.5`，这一位置就是 `5.5`。另一组权重会产生另一个输出通道。

写成下标形式是：

$$
Y_{n,o,i,j}
=b_o+
\sum_c\sum_u\sum_v
X_{n,c,iS_h+u-P_h,jS_w+v-P_w}W_{o,c,u,v}
$$

这里 `n` 是样本，`o` 和 `c` 分别是输出、输入通道，`i,j` 标出输出位置，`u,v` 则在卷积核内移动。`S_h,S_w` 表示 stride，`P_h,P_w` 表示 padding。

越过原图边界的位置按零处理。输出空间大小为：

$$
H_{out}=\left\lfloor\frac{H+2P_h-K_h}{S_h}\right\rfloor+1
$$

$$
W_{out}=\left\lfloor\frac{W+2P_w-K_w}{S_w}\right\rfloor+1
$$

![padding、stride 与输出大小](assets/padding_stride.png)

以 `H=W=5, K=3` 为例：

| padding | stride | 输出空间 |
| ---: | ---: | ---: |
| 0 | 1 | `3×3` |
| 1 | 1 | `5×5` |
| 0 | 2 | `2×2` |
| 1 | 2 | `3×3` |

## im2col 把每个窗口放成一行

直接照公式写，会对样本、输出通道、输出行和输出列反复循环。`im2col` 换了一个组织方式：先把所有窗口收集起来。下面仍用 `3×3` 核、stride 1、padding 0，不过输入缩成 `4×4`：

$$
\begin{bmatrix}
1&2&3&4\\
5&6&7&8\\
9&10&11&12\\
13&14&15&16
\end{bmatrix}
$$

窗口左上角只有四个合法位置。图中先框出这四个窗口，再把每个窗口按行展开：

![四个 3×3 窗口展开为 4×9 矩阵](assets/im2col_explained.png)

行的顺序是“输出第 1 行第 1 列、第 1 行第 2 列、第 2 行第 1 列、第 2 行第 2 列”：

$$
X_{col}=
\begin{bmatrix}
1&2&3&5&6&7&9&10&11\\
2&3&4&6&7&8&10&11&12\\
5&6&7&9&10&11&13&14&15\\
6&7&8&10&11&12&14&15&16
\end{bmatrix}
$$

所以 `X_col` 是 `(4,9)`，不是 `(9,4)`。把前面的核按行展开为 `(9,1)`：

```text
(4,9) @ (9,1) -> (4,1)
```

四行的结果都是 `−6`，再按输出位置排回去：

$$
\begin{bmatrix}
-6&-6\\
-6&-6
\end{bmatrix}
$$

多通道时，一行会先放入第 0 个通道的 `K_h×K_w` 个数，再接上第 1 个通道，直到 `C_in` 个通道全部放完。权重也按同样顺序展开，因而一个窗口的长度是 `C_in*K_h*K_w`：

```text
X_col  (N*H_out*W_out, C_in*K_h*K_w)
W_col  (C_in*K_h*K_w, C_out)
Y_col  (N*H_out*W_out, C_out)
```

$$
Y_{col}=X_{col}W_{col}+b
$$

矩阵乘法之后还有一步容易写错。`Y_col` 的每一行属于一个输出位置，同一行的各列属于不同输出通道；先恢复位置，再把通道轴搬到前面，才能回到 NCHW：

```python
Y = Y_col.reshape(N, H_out, W_out, C_out).transpose(0, 3, 1, 2)
```

假如四个位置、两个输出通道得到 `[[1,10],[2,20],[3,30],[4,40]]`，第一个通道应是 `[[1,2],[3,4]]`，第二个是 `[[10,20],[30,40]]`。直接 reshape 成 `(N,C_out,H_out,W_out)` 会把 1 和 10 排进同一个通道，单通道例子却看不出这个错误。

`im2col` 的加速来自把计算交给矩阵乘法，但展开的窗口需要额外内存。stride 1、padding 1 的 `3×3` 卷积会收集大约原输入九倍的元素，而且反向时还需要窗口梯度。小 batch 有助于控制这部分内存；真实框架可以采用不显式存下整个展开矩阵的实现。

## 用两行数字看梯度怎样累加

这里的矩阵公式和 Block 1 的线性层相同。先用两个长度为 2 的窗口代替真实卷积窗口：

```text
X_col = [[1,2],       W_col = [[3],       Y_col = [[11],
         [3,4]]                [4]]                [25]]
```

假设两个输出收到的上游梯度分别是 2 和 1，即 `dY_col=[[2],[1]]`。同一组权重被两个窗口共享，所以它的梯度包含两行贡献：

```text
dW_col = [[1×2 + 3×1],       = [[5],
          [2×2 + 4×1]]          [8]]
```

每行输入的梯度则是该行的上游梯度乘上权重：

```text
dX_col = [[2×3, 2×4],       = [[6,8],
          [1×3, 1×4]]          [3,4]]
```

偏置在两行前向中也被共享，因而这个例子里 `db=2+1=3`。换回全部卷积窗口，公式不变：

$$
dW_{col}=X_{col}^{\mathsf T}dY_{col}
$$

$$
dX_{col}=dY_{col}W_{col}^{\mathsf T}
$$

$$
db=\sum_{n,i,j}dY_{n,:,i,j}
$$

`dW` 会自动累加所有窗口的贡献。`dX_col` 还不能直接 reshape 成输入，因为相邻窗口中有重复像素。

这里不再除以窗口数或 batch size。如果上游交叉熵对 batch 取了平均，传进来的 `dY` 已经包含那次平均；卷积反向只按链式法则累加每条路径。每层都再除一次，会让深层网络的梯度越来越小。

## col2im 把重叠贡献加回原图

输入中心的像素会出现在多个窗口里，`im2col` 因此复制了它。反向时，这些副本收到的梯度都属于同一个原像素，`col2im()` 要把它们相加：

```text
col2im(im2col(X)) == X * coverage
```

这里的 `coverage` 是每个位置被窗口覆盖的次数。上面 `4×4` 输入的四个 `3×3` 窗口，覆盖次数为：

```text
[[1,2,2,1],
 [2,4,4,2],
 [2,4,4,2],
 [1,2,2,1]]
```

假如 `dX_col` 每个元素都是 1，`col2im` 返回的恰好就是这张表。中心像素在四个窗口里出现，便收到四份梯度；不能再除以 4，否则求出的就不是原来运算的导数了。

<div class="widget-mount" data-widget="im2col-grad"></div>

## 代码里最容易漏掉的一处

`Conv2D.forward()` 缓存输入、`X_col` 和输出 shape；`backward()` 用上面的矩阵公式得到 `dX、dW、db`。这里的梯度数组必须原位更新：

```python
self.dW[...] = computed_gradient
self.db[...] = computed_bias_gradient
```

若写成 `self.dW = computed_gradient`，优化器仍可能握着旧数组。此时打印新 `dW` 看似正常，`optimizer.step()` 却不会改变卷积权重。

为了确认反向公式，可以给输出一份固定的上游梯度 `G`，把整个层暂时当成标量函数 `L=sum(forward(X)*G)`。取某个权重 `w`，分别把它增加和减少 `ε` 后重新 forward：

$$
\frac{\partial L}{\partial w}\approx\frac{L(w+\varepsilon)-L(w-\varepsilon)}{2\varepsilon}
$$

这就是有限差分。测试使用 `float64`、`ε=1e-5`，把得到的数与 `backward(G)` 中对应位置的 `dW` 比较；对输入和偏置也能做同样的检查。`G` 固定以后，数值检查不依赖分类任务是否已经学会，因此卷积本身出错时更容易定位。

数值例子与有限差分检查可以一起运行：

```bash
python -m unittest discover -s tests -p 'test_block2.py' -v
python tests/test_im2col_figure.py -v
```

配图中的两组矩阵还由 [`scripts/validate_figure_content.py`](../../../scripts/validate_figure_content.py) 单独计算，避免图和代码各自维护一套数字。下一页把卷积接到 [BatchNorm 与 Global Average Pool](../task_12_pooling_and_bn/README.md)。

参考：[Dive into Deep Learning：图像卷积](https://d2l.ai/chapter_convolutional-neural-networks/conv-layer.html)、[CS231n 卷积网络](https://cs231n.github.io/convolutional-networks/)。
