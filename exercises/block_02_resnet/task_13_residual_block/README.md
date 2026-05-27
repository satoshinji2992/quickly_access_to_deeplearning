# task_13：残差块

前面我们已经把 CNN 需要的零件拆开写了：卷积负责提取局部特征，BatchNorm 负责稳定训练，ReLU 提供非线性，池化负责缩小空间尺寸。

现在问题变成：这些零件怎么拼成一个深一点、还能训练得动的网络？

如果只是简单堆很多层，直觉上模型应该更强。可实际训练时你会发现，深层普通 CNN 经常不如浅层网络。不是测试集差，而是训练集上也可能更差。这就说明问题不只是过拟合，而是优化本身变难了。

ResNet 的核心想法很朴素：让网络不要每一层都从零学完整变换，而是学一个“修正量”。

## 1. 这个任务解决什么问题

普通卷积块大概长这样：

```text
x -> Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> y
```

输入 `x` 必须一路经过所有变换，最后得到 `y`。如果层数很深，梯度也必须沿着这条长路反传回来。

残差块改成：

```text
x -> Conv -> BN -> ReLU -> Conv -> BN -> + -> ReLU -> y
|                                      ^
|______________________________________|
```

也就是：

```text
y = ReLU(F(x) + x)
```

这里的 `F(x)` 是卷积分支学到的东西。它不再被要求学完整输出，而是学“在原来的输入基础上应该改多少”。

这件事会带来两个好处：

- 如果某个阶段暂时不需要复杂变换，`F(x)` 可以接近 0，整个块就接近恒等映射。
- 反向传播时，梯度可以通过 shortcut 更直接地传回前面。

这就是 ResNet 能堆深的关键。

## 2. 如何让模型学会这个？

先别把残差连接想得太神秘。它其实像是在问模型：

“你真的要完全重写这份特征吗？还是只需要在原来的基础上改一点？”

假设前一层已经提取出了“有轮子”“有车窗”“大面积红色”这些特征。后面的层也许只需要把它们组合得更抽象，而不应该把这些信息全部洗掉。

普通网络里，每一层都可能把输入改得面目全非。残差块给输入留了一条直路：

```text
原信息 x
学习到的修正 F(x)
两者相加
```

如果卷积分支学得不好，shortcut 至少能保留原信息。如果卷积分支学得好，它就会在原信息上加上更有用的变化。

从梯度角度看，也很直观：

```text
y = F(x) + x
```

那么反向传播时：

```text
dL/dx = dL/dy * (dF/dx + 1)
```

这里的 `+ 1` 就是 shortcut 带来的直接通路。即使 `dF/dx` 很小，梯度也不至于完全消失。

当然，现实里还要处理一个问题：如果输入输出 shape 不一样，`x` 不能直接加到 `F(x)` 上。

比如：

```text
x:    (N, 16, 32, 32)
F(x): (N, 32, 16, 16)
```

通道数和空间尺寸都变了，不能直接相加。这个时候需要 projection shortcut，也就是用一个 `1x1` 卷积把 `x` 变成同样 shape：

```text
shortcut(x) = Conv1x1(x)
y = F(x) + shortcut(x)
```

这就是 `BasicBlock` 里 `needs_projection` 的意义。

## 3. 需要引入哪些数学对象

这一节的核心公式只有一个：

```text
out = ReLU(residual + shortcut)
```

其中主分支一般是：

```text
residual = BN2(Conv2(ReLU(BN1(Conv1(x)))))
```

完整写出来：

```text
z1 = Conv3x3(x)
z2 = BN(z1)
z3 = ReLU(z2)
z4 = Conv3x3(z3)
z5 = BN(z4)

if shape 不变:
    shortcut = x
else:
    shortcut = Conv1x1(x)

out = ReLU(z5 + shortcut)
```

shape 规则要看清楚。

当 `stride=1` 且 `in_channels == out_channels`：

```text
x:        (N, C, H, W)
residual: (N, C, H, W)
shortcut: x
out:      (N, C, H, W)
```

当 `stride=2` 或通道数变化：

```text
x:        (N, C_in, H, W)
residual: (N, C_out, H/2, W/2)
shortcut: (N, C_out, H/2, W/2)
out:      (N, C_out, H/2, W/2)
```

这里的 `H/2` 只是常见情况，实际由卷积输出尺寸公式决定。

反向传播要记住加法的规则：

```text
z = a + b
```

如果上游梯度是 `dz`，那么：

```text
da = dz
db = dz
```

所以残差块 backward 时，梯度要分成两份：

- 一份走主分支，依次传回 `BN2 -> Conv2 -> ReLU -> BN1 -> Conv1`。
- 一份走 shortcut，如果没有 projection 就直接加到输入梯度上；如果有 projection，就传过 `Conv1x1`。

最后两路梯度在输入处相加。

## 4. 对应代码该怎么组织

这一节主要看 [residual_block.py](/root/code/python/quickly_access_to_deeplearning/exercises/block_02_resnet/task_13_residual_block/residual_block.py)。

当前骨架是：

```python
class BasicBlock:
    """NumPy ResNet BasicBlock skeleton.

    Expected structure:
    Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> shortcut add -> ReLU
    """

    def __init__(self, in_channels, out_channels, stride=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.needs_projection = stride != 1 or in_channels != out_channels
```

你要在 `__init__` 里创建这些子层：

- 第一层 `3x3 Conv`，使用传入的 `stride`。
- 第一层 `BatchNorm2D`。
- 一个 ReLU。
- 第二层 `3x3 Conv`，通常 stride 为 1。
- 第二层 `BatchNorm2D`。
- 如果需要 projection，就创建一个 `1x1 Conv`。

如果你前面的 `Conv2D` 暂时只支持固定接口，那就先按你的实现来适配。重点不是名字完全一样，而是结构和数据流要对。

建议额外写一个很小的 ReLU 辅助类：

```python
class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask
```

这样主分支每一步都能像搭积木一样写：

```python
out = self.conv1.forward(x)
out = self.bn1.forward(out)
out = self.relu1.forward(out)
out = self.conv2.forward(out)
out = self.bn2.forward(out)
out = out + shortcut
out = self.relu_out.forward(out)
```

这比把 ReLU 写成散落的 `np.maximum` 更容易做 backward。

## 5. 逐步实现顺序

第一步，只处理最简单的情况：

```text
in_channels = out_channels
stride = 1
```

也就是 shortcut 直接等于 `x`。先让这个版本 forward 跑通。

输入：

```text
x.shape = (2, 16, 32, 32)
```

输出也应该是：

```text
out.shape = (2, 16, 32, 32)
```

第二步，加入 projection。

测试：

```text
x.shape = (2, 16, 32, 32)
out_channels = 32
stride = 2
```

输出通常应该接近：

```text
out.shape = (2, 32, 16, 16)
```

如果这一步 shape 对不上，不要继续写 backward。先把 forward 的 shape 全部打印出来：

```text
x
conv1
bn1
relu1
conv2
bn2
shortcut
add
out
```

第三步，写 backward 的最后一段。

因为最后是：

```text
out = ReLU(add)
add = residual + shortcut
```

所以先做：

```text
dadd = relu_out.backward(dout)
dresidual = dadd
dshortcut = dadd
```

第四步，主分支从后往前传：

```text
d = bn2.backward(dresidual)
d = conv2.backward(d)
d = relu1.backward(d)
d = bn1.backward(d)
d = conv1.backward(d)
```

第五步，shortcut 分支处理：

```text
if needs_projection:
    dx_shortcut = projection.backward(dshortcut)
else:
    dx_shortcut = dshortcut
```

第六步，两路输入梯度相加：

```text
dx = dx_main + dx_shortcut
```

这个 `+` 非常重要。很多初学者会只返回主分支的梯度，忘了 shortcut 对输入也有贡献。

## 6. 常见错误和调试方法

最常见错误是加法 shape 不一致。

残差连接不是“差不多能加就行”，而是每个维度都要完全一致。只要通道数或空间尺寸变了，就需要 projection。

第二个错误是 stride 放错位置。

BasicBlock 常见做法是在第一个 3x3 卷积里使用 stride，把空间尺寸降下来；第二个 3x3 卷积 stride 为 1。projection shortcut 也要用同样的 stride，这样两条分支的空间尺寸才能对齐。

第三个错误是 ReLU 的缓存被覆盖。

如果你复用同一个 ReLU 实例做两次 forward：

```python
self.relu.forward(...)
self.relu.forward(...)
```

第二次会覆盖第一次的 mask，backward 就错了。建议 `relu1` 和 `relu_out` 用两个实例。

第四个错误是忘记 shortcut 梯度。

如果没有 projection，shortcut 也是一条真实的数据路径。反向传播时它给输入的梯度就是 `dshortcut` 本身。

第五个错误是 BatchNorm 的 train/eval 状态没有递归设置。

以后整个 ResNet 切换训练和评估模式时，残差块内部的 BN 也要一起切换。不然验证结果会乱跳。

调试残差块时，推荐先把输入尺寸缩小：

```text
(2, 4, 8, 8)
```

不要一开始就用 `(64, 64, 32, 32)`。小张量更快，也更容易打印。

## 7. 必做实验

1. identity shortcut 测试。

   设置 `in_channels == out_channels` 且 `stride=1`，确认输出 shape 不变。

2. projection shortcut 测试。

   设置 `in_channels != out_channels` 或 `stride=2`，确认主分支和 shortcut 分支 shape 完全一致。

3. 零主分支实验。

   临时让主分支输出全 0，观察残差块是否近似变成 `ReLU(x)`。这个实验能帮你理解 shortcut 不是装饰品，而是真的在传信息。

4. 梯度分流实验。

   在 backward 里打印 `dx_main` 和 `dx_shortcut` 的范数，确认两条路径都有梯度。

5. 对比 plain block。

   写一个没有 shortcut 的普通块，用相同随机输入跑 forward/backward，比较梯度范数。你会更直观地看到 shortcut 让梯度多了一条路。

6. train/eval 状态检查。

   给 `BasicBlock` 加上 `train()` 和 `eval()` 时，要确认内部所有 BN 都跟着切换。

## 8. 完成后应该能回答的问题

完成这一节后，你应该能回答：

- ResNet 为什么不是简单地“层数更多”？
- `F(x) + x` 里的 `F(x)` 学的是什么？
- shortcut 为什么能帮助梯度传播？
- 什么时候可以直接用 identity shortcut？
- 什么时候必须用 projection shortcut？
- projection shortcut 为什么通常用 `1x1` 卷积？
- 残差块 backward 时，为什么输入梯度要来自两条路径相加？
- ReLU 的 mask 为什么不能被复用覆盖？
- BatchNorm 在残差块里通常放在哪些位置？

如果你能把这些讲清楚，ResNet 论文里最核心的直觉你已经吃下来了。

## 9. 下一任务为什么自然出现

现在你有了一个可以重复堆叠的 `BasicBlock`。单个残差块只是零件，真正的 ResNet 是把多个 block 按 stage 组织起来：

```text
stem -> stage1 -> stage2 -> stage3 -> global average pool -> classifier
```

每个 stage 里，空间尺寸通常逐渐变小，通道数逐渐变多。模型从低级边缘纹理，慢慢学到更抽象的类别特征。

下一节我们会把这些模块装成一个 NumPy ResNet，并且不急着直接冲完整训练。先做小样本过拟合测试：如果模型连 500 张图片都记不住，那说明代码还有问题；如果小样本能过拟合，再谈泛化才有意义。
