# task_14: NumPy ResNet 训练

前面几关你已经把零件做出来了.

现在要把它们装成一个能训练的图像分类模型.

这一步通常比想象中烦. 因为单个层能跑, 不代表整个训练过程就对. 数据增强、BatchNorm 模式、学习率、loss、checkpoint, 任何一个地方写错, 模型都可能学不动.

所以这一关不要一上来追准确率.

先让模型在小数据上过拟合.

---

## 一. 先别急着跑完整 CIFAR-100

很多人写完模型后第一反应是: 直接训练完整数据集.

然后发现 loss 不降, accuracy 像随机猜.

这时候你不知道问题在哪里. 可能是卷积 backward 错了, 可能是 BN eval 模式错了, 可能是标签打乱了, 也可能只是学习率太大.

更好的办法是先做小样本过拟合:

```text
拿 100 到 500 张图片
关闭或减少随机增强
训练很多轮
看 train loss 能不能明显下降
看 train acc 能不能接近 100%
```

如果小样本都过拟合不了, 说明模型或训练循环有 bug.

不要跳过这一步. 它能省很多时间.

---

## 二. 一个轻量 ResNet 长什么样?

我们不需要一开始就复刻完整 ResNet-18.

在 CIFAR 这种小图上, 可以用一个更轻量的结构:

```text
input: (N, 3, 32, 32)

Conv3x3 -> BN -> ReLU

stage1: BasicBlock, BasicBlock
stage2: BasicBlock(stride=2), BasicBlock
stage3: BasicBlock(stride=2), BasicBlock

GlobalAvgPool
Linear -> 100 logits
```

这里的 100 是 CIFAR-100 的类别数.

每个 stage 开始时可以用 stride=2 降采样, 空间尺寸从 32 变到 16, 再变到 8. 通道数也可以逐渐变大, 比如:

```text
16 -> 32 -> 64
```

这不是唯一写法. 但先用一个小模型跑通, 比一开始堆很深更靠谱.

---

## 三. 训练循环还是那四步

虽然现在模型复杂了, 训练循环本质还是 Block 1 那几步:

```text
forward -> loss -> backward -> update
```

只是里面多了几个容易忘的细节.

训练模式:

```python
model.train()
```

BatchNorm 使用当前 batch 的统计量, 并更新 running mean/var.

验证模式:

```python
model.eval()
```

BatchNorm 使用 running mean/var, 不再使用当前 batch 统计量.

如果你忘了切换, 验证准确率会飘得很厉害.

---

## 四. 学习率不要乱猜太久

自己写 NumPy 训练 ResNet, 不要一开始就追求漂亮曲线.

先用几个保守设置:

- optimizer: Momentum SGD 或 AdamW.
- batch size: 32 或 64.
- learning rate: 从小一点开始试, 比如 0.01.
- weight decay: 可以先不用, 跑通后再加.

如果 loss 直接变成 NaN, 先降学习率.

如果 loss 几乎不动, 先看梯度范数和参数有没有更新.

如果训练集能上升, 验证集很差, 再考虑数据增强、正则化和模型大小.

不要一次改太多东西. 不然你不知道到底是谁起作用.

---

## 五. checkpoint 和日志

训练图像模型很慢, 哪怕是小模型.

所以你需要保存:

- 当前 epoch.
- 模型参数.
- optimizer 状态.
- train loss / train acc.
- val loss / val acc.

日志不用花哨. 一行能看懂就行:

```text
epoch=10 train_loss=1.82 train_acc=0.46 val_loss=2.30 val_acc=0.31
```

如果训练中断, checkpoint 可以让你继续. 如果结果变差, 日志可以让你回头看是哪次改动开始出问题.

这也是为什么后面单独有一个实验记录任务. 不是让你写报告, 是为了让你别忘了自己做过什么.

---

## 六. 你要完成什么?

请在 `train_resnet.py` 中组装一个轻量 ResNet 训练流程.

当前文件还只是占位:

```python
print("TODO: assemble your NumPy ResNet and train CIFAR-100.")
print("Start with a 500-image overfit test before full training.")
```

你可以按这个顺序做:

1. 先导入前面几关的层: Conv2D、BatchNorm2D、BasicBlock、GlobalAvgPool2D.
2. 写一个小的 ResNet 类, 提供 `forward`、`backward`、`parameters`、`train`、`eval`.
3. 接入 task_10 的数据管线.
4. 先跑 100 到 500 张图片的小样本过拟合.
5. 小样本能学住以后, 再跑更多数据.
6. 保存日志和 checkpoint.

如果完整 CIFAR-100 训练太慢, 没关系. 先把小样本过拟合跑通, 这比一个随机准确率的大训练更有价值.

下一关不要求写正式报告, 只是把关键现象记下来.
