# task_11：Conv2D 与 im2col

现在数据已经能以 `NCHW` 的形式进入模型了。下一步，我们要写 CNN 里最核心的层：卷积。

卷积这件事看起来很像魔法：一个小核在图片上滑一滑，模型就能学边缘、纹理、形状。可真正写代码时，它就是一堆非常具体的数组索引、矩阵乘法和 shape 变换。

这一节的目标不是背“卷积可以提特征”，而是把 `Conv2D` 的 forward 和 backward 写到你能逐行解释。

## 1. 这个任务解决什么问题

图片有局部结构。一个边缘、一个角点、一小块纹理，通常只和附近像素有关。

MLP 会把整张图摊平成向量，一上来就把所有像素混在一起。卷积不是这样。卷积拿一个小窗口，比如 `3 x 3`，在图片上滑动，用同一组权重去扫描每个位置。

这个任务要解决的是：

```text
怎么在 NumPy 里实现一个可以 forward 和 backward 的卷积层？
```

你要写的不只是输出，还包括反向传播。因为只会 forward 的卷积只能做推理，不能训练。

## 2. 如何让模型学会这个？

卷积层学的是一组卷积核。一个卷积核可以看成一个小模板。训练开始时它是随机的，训练过程中它会慢慢变成能检测某种局部模式的东西。

早期卷积层可能学到：

- 横向边缘
- 纵向边缘
- 颜色块
- 简单纹理

更深的层会组合这些局部模式，比如“轮子”“眼睛”“叶子边缘”。

如果我们直接用循环写卷积，大概会有五层循环：

```text
样本 -> 输出通道 -> 高度位置 -> 宽度位置 -> 小窗口乘加
```

这能写，但很慢，也很容易错。

`im2col` 的想法是：把每个滑动窗口摊平成一行。

比如一张单通道 `4 x 4` 图片，用 `2 x 2` 窗口、stride 为 1，会得到 9 个窗口。每个窗口摊平成长度为 4 的向量，最后得到：

```text
cols.shape = (9, 4)
```

卷积核也摊平成一列或几列。于是很多小窗口的乘加，就变成了一次矩阵乘法。

这就是深度学习框架常做的事：把看起来特殊的运算，改写成硬件很擅长的矩阵运算。

## 3. 需要引入哪些数学对象

输入：

```text
X in R^(N x C_in x H x W)
```

卷积核：

```text
W in R^(C_out x C_in x K x K)
```

偏置：

```text
b in R^(C_out)
```

输出空间大小：

```text
H_out = (H + 2P - K) // S + 1
W_out = (W + 2P - K) // S + 1
```

其中：

- `P` 是 padding
- `S` 是 stride
- `K` 是 kernel size

`im2col` 后：

```text
cols.shape = (N * H_out * W_out, C_in * K * K)
```

卷积核摊平后：

```text
w_col.shape = (C_in * K * K, C_out)
```

于是：

```text
out_col = cols @ w_col + b
```

再 reshape 回：

```text
out.shape = (N, C_out, H_out, W_out)
```

反向传播也可以沿着这个矩阵乘法来理解。

如果：

```text
out_col = cols @ w_col + b
```

那么：

```text
dcols = dout_col @ w_col.T
dw_col = cols.T @ dout_col
db = sum(dout_col, axis=0)
```

最后：

```text
dW = dw_col reshape 回 (C_out, C_in, K, K)
dx = col2im(dcols)
```

这里 `col2im` 很重要。因为输入里的同一个像素可能出现在多个滑动窗口里，反向时这些窗口传回来的梯度要累加。

## 4. 对应代码该怎么组织

核心文件是 [conv2d.py](/root/code/python/quickly_access_to_deeplearning/exercises/block_02_resnet/task_11_conv2d_im2col/conv2d.py)。

你需要补：

```python
compute_output_size(...)
im2col(...)
col2im(...)
Conv2D.forward(...)
Conv2D.backward(...)
```

`Conv2D.forward` 的主流程应该像这样：

```text
保存输入 shape
用 im2col 展开输入窗口
把权重 reshape 成矩阵
矩阵乘法得到 out_col
加 bias
reshape 成 NCHW 输出
```

`Conv2D.backward` 的主流程应该像这样：

```text
把 dout 转成 dout_col
计算 db
计算 dW
计算 dcols
用 col2im 折回 dx
```

建议把 `im2col` 和 `col2im` 写成独立函数，不要藏在 `Conv2D` 里面。这样你可以单独测试它们。

## 5. 逐步实现顺序

第一步，确认输出尺寸公式。

```python
compute_output_size(32, kernel_size=3, stride=1, padding=1) == 32
compute_output_size(32, kernel_size=3, stride=2, padding=1) == 16
```

如果这里错了，后面所有 shape 都会错。

第二步，写 `im2col`。

用一个非常小的输入测试：

```python
x = np.arange(1 * 1 * 4 * 4).reshape(1, 1, 4, 4)
cols = im2col(x, kernel_size=2, stride=1, padding=0)
print(cols)
```

你应该能肉眼看出每一行是不是一个 `2 x 2` 小窗口。

第三步，加 padding。

先不要一开始就写复杂。`padding=0` 跑通后，再测试 `padding=1`。padding 后的图片边缘应该补 0。

第四步，写 `Conv2D.forward`。

用随机输入确认输出 shape：

```python
x = np.random.randn(2, 3, 32, 32)
conv = Conv2D(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
out = conv.forward(x)
print(out.shape)  # (2, 16, 32, 32)
```

第五步，写 `col2im`。

`col2im` 不是 `im2col` 的简单反操作。因为多个窗口会覆盖同一个输入像素，所以折回时要累加。

可以用全 1 的 `cols` 测试，观察中间像素被覆盖次数更多，边缘像素被覆盖次数更少。

第六步，写 `Conv2D.backward`。

先确认三个梯度的 shape：

```text
dW.shape == W.shape
db.shape == b.shape
dx.shape == x.shape
```

第七步，做数值梯度检查。

不要用大输入。可以用：

```text
x.shape = (2, 2, 5, 5)
out_channels = 3
kernel_size = 3
padding = 1
```

只检查几个随机位置也可以。目标是确认反向传播没有方向性错误。

## 6. 常见错误和调试方法

输出尺寸不对：先查公式 `(H + 2P - K) // S + 1`。如果不能整除，要明确你的实现是否允许这种情况。

通道顺序错：本教程统一使用 `NCHW`。如果你传进来的是 `NHWC`，卷积会把高度当通道，结果一定乱。

`im2col` 行顺序混乱：只要 forward 和 backward 保持同一种顺序，数学上可以工作。但为了调试，建议固定成 `N -> H_out -> W_out` 的顺序。

`col2im` 梯度太大或太小：检查重叠区域是否累加，而不是覆盖。NumPy 里可以用 `np.add.at` 处理重复索引累加。

`dW` shape 错：卷积核梯度最后必须 reshape 回 `(out_channels, in_channels, K, K)`。

`db` 忘记对 batch 和空间维求和：偏置对每个输出通道只有一个数，所以 `db` 应该是 `(out_channels,)`。

forward 能跑，backward 不对：做数值梯度检查。不要跳过。卷积 backward 很容易看起来合理，实际错一个 transpose。

## 7. 必做实验

1. 小输入肉眼检查。

   用 `1 x 1 x 4 x 4` 输入，`kernel_size=2`，打印 `im2col`，确认每一行是正确窗口。

2. padding 检查。

   用全 1 输入加 `padding=1`，确认边缘补的是 0，输出尺寸符合公式。

3. shape 检查。

   设置 `N=2, C=3, H=32, W=32, K=3, P=1`，输出应该是 `(2, out_channels, 32, 32)`。

4. stride 检查。

   设置 `stride=2`，观察输出空间尺寸是否变小。

5. col2im 覆盖次数检查。

   用全 1 的 `cols` 折回，看中心区域数值是否比边缘大。这能证明你在累加，而不是覆盖。

6. 梯度检查。

   对 `W`、`b`、`x` 分别做 numerical check。误差不需要完美到机器精度，但应该足够小。

7. 与朴素卷积对比。

   写一个很慢但直观的四重循环卷积，只用于小输入。比较它和 `im2col` 版本输出是否一致。

## 8. 完成后应该能回答的问题

完成这一节后，你应该能回答：

- 卷积核的 shape 为什么是 `(out_c, in_c, k, k)`？
- `out_channels` 表示什么？
- padding 和 stride 分别改变了什么？
- `im2col` 为什么能把卷积变成矩阵乘法？
- `cols` 的每一行代表什么？
- `col2im` 为什么需要累加？
- `dW`、`db`、`dx` 分别代表什么？
- 为什么卷积比 MLP 更适合图片？
- 为什么数值梯度检查对卷积尤其重要？

如果这些问题能讲清楚，你就不只是“调通了卷积”，而是真的理解了 CNN 的第一块地基。

## 9. 下一任务为什么自然出现

现在你有了卷积层。

但 CNN 不只靠卷积。我们还需要降采样，让空间尺寸变小；需要 BatchNorm，让训练更稳；最后还需要把空间信息汇聚成分类向量。

下一任务就补这些层。写完它们之后，你才有足够零件去搭一个残差块。
