# task_10：图像数据管线

进入 ResNet 之前，先别急着写网络。

很多初学者第一次做图像分类，会把注意力全放在模型结构上：卷积几层、通道多少、要不要残差。可训练真正开始后，最先出问题的往往不是模型，而是数据。

图片有没有转成正确的维度？像素值有没有缩放？增强有没有只在训练集上做？标签和图片 shuffle 后还对得上吗？这些问题只要错一个，后面的 ResNet 写得再漂亮也学不起来。

这一节先把图像数据管线理顺。

## 1. 这个任务解决什么问题

上一块的 MNIST MLP 把图片拉平成了 `784` 维向量。现在我们要正式进入 CNN/ResNet，就不能再把图片当普通表格数据了。

CIFAR-100 的每张图片是：

```text
32 x 32 x 3
```

这里的 3 是 RGB 三个颜色通道。你要训练图像模型，第一步不是写 ResNet，而是先把图片变成训练循环能稳定使用的 batch。

这个任务解决的是：

```text
原始图片和标签 -> 预处理 -> 数据增强 -> mini-batch -> 模型输入
```

如果这个环节可靠，后面模型出问题时，你至少可以排除“数据一开始就是乱的”。

## 2. 如何让模型学会这个？

模型不能直接从“文件夹里的一堆图片”学习。它需要一批一批的数值张量。

对 CNN 来说，我们希望输入长这样：

```text
(N, C, H, W)
```

也就是：

- `N`：batch size
- `C`：通道数，RGB 图片是 3
- `H`：高度
- `W`：宽度

很多图片库读出来是 `NHWC`，也就是：

```text
(N, H, W, C)
```

比如一批 CIFAR-100 图片可能是：

```text
(128, 32, 32, 3)
```

但我们后面手写 `Conv2D` 时会采用 `NCHW`：

```text
(128, 3, 32, 32)
```

所以第一步是转置。

然后要标准化。图片像素原本是 0 到 255，数值范围太大。我们通常先除以 255，把范围变成 0 到 1，再减去均值、除以标准差。

这件事和任务一里的特征标准化是同一个思想：不要让输入尺度给梯度下降添乱。

最后是数据增强。训练时随机裁剪、随机水平翻转，不是为了花哨，而是告诉模型：

```text
物体稍微平移一点，类别不应该变。
物体水平翻转后，很多类别仍然不变。
```

注意，这句话里有个“很多”。不是所有任务都适合水平翻转。比如识别文字、交通标志方向时，翻转可能改变含义。但 CIFAR-100 的大多数自然物体分类可以使用水平翻转。

## 3. 需要引入哪些数学对象

原始图片：

```text
X in R^(N x H x W x C)
```

转成 `NCHW`：

```text
X in R^(N x C x H x W)
```

在 NumPy 里就是：

```python
x = images.transpose(0, 3, 1, 2)
```

标准化：

```text
X_norm = (X / 255 - mean) / std
```

其中 `mean` 和 `std` 是每个通道的均值和标准差。为了和 `NCHW` 广播，它们最好变成：

```python
(1, 3, 1, 1)
```

这样每个通道都会用自己的均值和标准差：

```text
R 通道用 mean[0], std[0]
G 通道用 mean[1], std[1]
B 通道用 mean[2], std[2]
```

随机水平翻转是对宽度维做反转：

```python
x[:, :, :, ::-1]
```

随机裁剪通常是先 padding，再裁回原尺寸。CIFAR-100 常见做法是：

```text
32 x 32 -> 四周补 4 个像素 -> 40 x 40 -> 随机裁 32 x 32
```

mini-batch 生成则是把索引切成一段一段：

```text
indices = shuffle(0, 1, 2, ..., N-1)
batch_indices = indices[start:start+batch_size]
```

只要图片和标签使用同一组 `batch_indices`，它们就不会错位。

## 4. 对应代码该怎么组织

本任务核心文件是 [data_pipeline.py](/root/code/python/quickly_access_to_deeplearning/exercises/block_02_resnet/task_10_image_data_pipeline/data_pipeline.py)。

里面有这些函数：

```python
to_nchw(images)
normalize(images, mean, std)
random_horizontal_flip(images, p=0.5, rng=None)
random_crop_with_padding(images, padding=4, crop_size=32, rng=None)
iterate_minibatches(images, labels, batch_size, shuffle=True, seed=None)
```

你需要重点读懂每个函数的输入输出 shape。

建议你在每个函数下面都写一两个小测试，而不是直接接到训练里：

```python
images = np.random.randint(0, 256, size=(8, 32, 32, 3), dtype=np.uint8)
x = to_nchw(images)
assert x.shape == (8, 3, 32, 32)
```

图像任务里，shape 比很多公式更重要。只要 shape 脑子里清楚，后面卷积、池化、残差块都会顺很多。

## 5. 逐步实现顺序

第一步，准备一小批假图片测试。

```python
images = np.random.randint(0, 256, size=(8, 32, 32, 3), dtype=np.uint8)
labels = np.arange(8)
```

先用假数据，因为它快，而且不会被数据集下载、读取格式这些外部问题干扰。

第二步，转成 NCHW。

```python
x = to_nchw(images)
print(x.shape)  # (8, 3, 32, 32)
```

第三步，做标准化。

```python
mean = [0.5071, 0.4867, 0.4408]
std = [0.2675, 0.2565, 0.2761]
x = normalize(x, mean, std)
```

标准化后可以打印每个通道的均值和方差。它不一定刚好是 0 和 1，因为你只用了 8 张假图片，但数量级应该合理。

第四步，做随机裁剪。

先固定 seed：

```python
rng = np.random.default_rng(42)
x_aug = random_crop_with_padding(x, padding=4, crop_size=32, rng=rng)
```

确认输出还是：

```text
(8, 3, 32, 32)
```

第五步，做随机水平翻转。

同样固定 seed，观察同一个 seed 是否得到同样结果。可复现是调试的朋友。

第六步，用 `iterate_minibatches` 生成 batch。

```python
for xb, yb in iterate_minibatches(x, labels, batch_size=4, shuffle=True, seed=0):
    print(xb.shape, yb)
```

确认每个 batch 的图片和标签数量一致。

第七步，可视化增强前后的图片。

注意可视化前要转回 `HWC`：

```python
img = x[0].transpose(1, 2, 0)
```

如果图片颜色非常怪，通常就是通道顺序或反标准化出错了。

## 6. 常见错误和调试方法

颜色不对：大概率是 `NCHW` 和 `NHWC` 搞混了。`matplotlib.imshow` 通常需要 `HWC`。

shape 不对：每一步都打印 shape。尤其是 crop 和 flip 之后。不要只看最后报错，越早发现越便宜。

训练不稳定：检查是否除以 255，是否用了合理的 mean/std。如果输入还是 0 到 255，后面的梯度会很难调。

增强太强：CIFAR-100 图片只有 `32 x 32`，别做太激进的裁剪。主体本来就小，裁太狠可能把关键部分裁没。

每次结果不同：传入固定 seed 的 `rng`，先保证可复现，再做随机实验。

标签错位：shuffle 时必须用同一组索引取图片和标签。调试时可以先不 shuffle，确认第 `i` 张图片和 `labels[i]` 能对上。

验证集被增强：验证集通常只做确定性的预处理，不做随机裁剪和随机翻转。否则同一个模型每次验证结果都可能不同。

## 7. 必做实验

1. 原图 batch。

   不增强，只转 `NCHW`，打印 shape，并可视化 8 张图片。

2. 标准化检查。

   标准化前后分别打印 `min/max/mean/std`。确认像素不再是 0 到 255。

3. 随机翻转检查。

   设置 `p=1.0`，确认每张图都被水平翻转；再设置 `p=0.0`，确认完全不变。

4. 随机裁剪检查。

   设置 `padding=4`，输出尺寸应该保持 `32 x 32`。可视化几张增强后的图。

5. 固定 seed。

   同一个 seed 跑两次，结果应该相同；不同 seed 跑两次，结果可以不同。

6. mini-batch 检查。

   使用 `batch_size=3` 处理 8 张图片，观察最后一个 batch 怎么处理。你要明确自己的实现是丢弃最后不足 batch，还是保留。

7. 训练/验证预处理对比。

   写下训练集和验证集分别用了哪些处理。训练可以随机增强，验证应该稳定。

## 8. 完成后应该能回答的问题

完成这一节后，你应该能回答：

- `NCHW` 和 `NHWC` 有什么区别？
- 为什么本教程后面的卷积采用 `NCHW`？
- 标准化里的 mean/std 是按什么维度来的？
- 为什么图片像素通常要先除以 255？
- 随机裁剪和水平翻转为什么能提升泛化？
- 什么情况下水平翻转可能不合适？
- 为什么可视化前要转回 `HWC`？
- mini-batch 生成时为什么要 shuffle？
- 为什么验证集不应该使用随机增强？
- 固定 seed 对调试有什么帮助？

这些问题答清楚，后面训练 ResNet 时你就不会把所有锅都甩给模型结构。

## 9. 下一任务为什么自然出现

现在图片可以稳定进入训练循环了。

但模型还不知道怎么利用图片结构。下一步我们写卷积。卷积会用一个小窗口在图片上滑动，寻找边缘、纹理和局部形状。

如果直接用多层 Python 循环写卷积，它会慢，也很难写对。下一节会先实现 `im2col`，把卷积变成矩阵乘法。你会第一次看到：很多深度学习算子看起来复杂，底层其实是在精心组织矩阵乘法。
