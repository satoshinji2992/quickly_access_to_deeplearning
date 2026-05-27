# task_14：NumPy ResNet 训练

到这里，你已经写了很多“层”。但深度学习不是把层写出来就结束了。真正的模型训练，是把数据、模型、loss、优化器、日志、验证、保存和恢复都接起来。

这一节的目标不是追求很高的 CIFAR-100 准确率。用纯 NumPy 训练 ResNet，本来就会慢。我们的目标是让你完整经历一次研究者会做的训练流程：先验证代码能学习，再扩大数据，再记录实验。

## 1. 这个任务解决什么问题

前面的任务分别解决了局部问题：

- task_10：图片数据怎么读、怎么增强、怎么分 batch。
- task_11：卷积层怎么前向和反向。
- task_12：池化和 BatchNorm 怎么稳定训练。
- task_13：残差块怎么让深层网络更容易优化。

现在要把它们合成一个系统：

```text
images -> data pipeline -> ResNet -> logits -> loss -> backward -> optimizer -> metrics
```

这个系统如果能跑通，你就不再只是“实现了几个层”，而是写出了一个可以训练、可以评估、可以做对照实验的小型图像分类框架。

这一步很重要。很多 bug 不会出现在单层测试里，只会出现在完整训练时：

- loss 不下降。
- accuracy 永远接近随机猜。
- BatchNorm 训练正常，验证崩掉。
- 小样本都不能过拟合。
- checkpoint 恢复后指标突然变差。

这些问题就是训练系统要处理的。

## 2. 如何让模型学会这个？

训练一个图像分类模型，本质上还是任务一那件事：

```text
预测 -> 计算错误 -> 求梯度 -> 更新参数
```

只是现在模型更深，数据更复杂，参数更多。

每张图片进入 ResNet 后，模型输出一个长度为 100 的向量：

```text
logits.shape = (N, 100)
```

每个位置对应 CIFAR-100 的一个类别。这个向量还不是概率，它只是模型给每个类别的“分数”。接着用 softmax 变成概率，再用 cross entropy 衡量预测和真实标签的距离。

如果模型把“猫”预测成“车”，loss 会大；反向传播会告诉最后的分类器、前面的残差块、再前面的卷积层：你们刚才哪些激活让错误类别分数太高了，哪些参数该调整。

但是直接上完整训练不是好习惯。

研究者训练模型时，通常会先做一个非常重要的小实验：小样本过拟合。

拿 500 张图片，反复训练。如果代码正确、模型容量够，训练准确率应该明显上升，甚至能接近记住这 500 张图。

如果 500 张都记不住，先别谈泛化，先查代码。

这个检查很朴素，但非常有用。

## 3. 需要引入哪些数学对象

这一节会用到几个训练级别的对象。

第一，logits 和 softmax。

```text
logits = model(x)          # (N, 100)
prob = softmax(logits)     # (N, 100)
```

softmax：

```text
prob[i, k] = exp(logits[i, k]) / sum_j exp(logits[i, j])
```

实际代码里要做数值稳定：

```python
shifted = logits - np.max(logits, axis=1, keepdims=True)
exp = np.exp(shifted)
prob = exp / np.sum(exp, axis=1, keepdims=True)
```

第二，交叉熵。

如果标签是整数类别 `y[i]`：

```text
loss = -mean(log(prob[i, y[i]]))
```

softmax + cross entropy 的梯度可以写得很简洁：

```python
dlogits = prob.copy()
dlogits[np.arange(N), y] -= 1
dlogits /= N
```

第三，准确率。

```python
pred = np.argmax(logits, axis=1)
acc = np.mean(pred == y)
```

第四，参数更新。

最简单的 SGD：

```text
param = param - lr * grad
```

如果你在 Block 1 的小型库里已经写过 Momentum，也可以复用：

```text
v = momentum * v - lr * grad
param = param + v
```

第五，训练和验证模式。

有 BatchNorm 后，这件事非常重要：

```python
model.train()
# 训练：BN 使用当前 batch 统计量，并更新 running stats

model.eval()
# 验证：BN 使用 running stats
```

如果忘了切模式，验证指标会很难解释。

## 4. 对应代码该怎么组织

这一节入口是 [train_resnet.py](/root/code/python/quickly_access_to_deeplearning/exercises/block_02_resnet/task_14_numpy_resnet_train/train_resnet.py)。

现在它还是一个占位脚本：

```python
def main():
    print("TODO: assemble your NumPy ResNet and train CIFAR-100.")
    print("Start with a 500-image overfit test before full training.")
```

你可以按下面的结构扩展：

```text
train_resnet.py
├── parse_args()
├── set_seed()
├── load_cifar100()
├── build_model()
├── cross_entropy_loss()
├── train_one_epoch()
├── evaluate()
├── save_checkpoint()
├── load_checkpoint()
└── main()
```

模型结构可以先做轻量版，不要一开始就追求论文里的 ResNet-18。纯 NumPy 训练速度有限，教学目标是跑通完整系统。

一个可行的轻量结构：

```text
Conv3x3(3 -> 16)
BN
ReLU

Stage 1: BasicBlock(16 -> 16, stride=1) x 2
Stage 2: BasicBlock(16 -> 32, stride=2) x 2
Stage 3: BasicBlock(32 -> 64, stride=2) x 2

GlobalAvgPool2D
Linear(64 -> 100)
```

你也可以先把每个 stage 的 block 数量设成 1，确认能跑，再加深。

训练日志至少记录：

```text
epoch
train loss
train acc
val loss
val acc
learning rate
time
```

checkpoint 至少保存：

```text
model parameters
optimizer state
epoch
best val acc
config
```

不要只保存权重。实验复现时，配置同样重要。

## 5. 逐步实现顺序

第一步，先让模型能做一次 forward。

用随机输入：

```python
x = np.random.randn(4, 3, 32, 32)
logits = model.forward(x)
print(logits.shape)
```

输出应该是：

```text
(4, 100)
```

如果这一步不对，先不要读数据集。随机数据能排除 data pipeline 的干扰。

第二步，接上 cross entropy。

```python
y = np.array([0, 1, 2, 3])
loss, dlogits = cross_entropy_loss(logits, y)
```

确认：

```text
loss 是一个标量
dlogits.shape == logits.shape
```

第三步，跑一次 backward。

让梯度从 `dlogits` 传回模型所有层。此时你要检查每个参数都有对应梯度，shape 一致。

第四步，做一轮参数更新。

先用 SGD，不要一开始就把优化器写复杂。只要确认参数会变，loss 能重新计算。

第五步，小样本过拟合。

取 500 张训练图片，关闭或减少强数据增强，训练很多轮。你希望看到：

```text
train loss 下降
train acc 上升
```

如果 train acc 长期停在 1% 左右，说明模型和随机猜差不多。CIFAR-100 有 100 类，随机猜大约就是 1%。

第六步，完整训练。

小样本能过拟合后，再打开数据增强，跑完整训练集和验证集。

第七步，保存日志和 checkpoint。

每个实验都应该留下痕迹。否则你后面写报告时，只能凭印象说“好像变好了”，这不是研究。

## 6. 常见错误和调试方法

最常见问题是 loss 不下降。

排查顺序建议固定下来：

1. 先看 softmax + cross entropy 是否正确。
2. 再看最后一层 Linear 的梯度是否正确。
3. 再看模型能否在 10 张图片上过拟合。
4. 再扩大到 500 张。
5. 最后再碰完整训练。

第二个问题是 learning rate 不合适。

学习率太大，loss 会 NaN 或剧烈震荡；学习率太小，loss 几乎不动。可以先试：

```text
1e-1, 1e-2, 1e-3
```

观察前几十个 step 的 loss。

第三个问题是 BatchNorm 模式错了。

训练时忘了 `train()`，running stats 不更新；验证时忘了 `eval()`，指标依赖当前 batch。这个问题很隐蔽，因为代码不会报错，只是结果怪。

第四个问题是数据增强太强，导致小样本过拟合失败。

小样本过拟合测试的目标是证明模型能记住训练数据，所以可以先关掉随机裁剪、翻转。等能过拟合后，再打开增强做泛化实验。

第五个问题是标签和图片打乱不同步。

如果图片 shuffle 了，标签没跟着一起 shuffle，模型永远学不到。每个 batch 打印几张图片和标签，是非常值得做的检查。

第六个问题是 checkpoint 只保存了部分参数。

恢复训练后如果 loss 突然不连续，检查 optimizer state、BN running stats 有没有一起保存。

## 7. 必做实验

1. 随机输入 forward。

   用 `(4, 3, 32, 32)` 随机输入，确认输出是 `(4, 100)`。

2. 10 张图片过拟合。

   取 10 张图片训练到很高 train acc。这个实验主要检查梯度链路有没有断。

3. 500 张图片过拟合。

   这是正式进入完整训练前的门槛。记录 loss 和 acc 曲线。

4. learning rate 扫描。

   至少比较 `1e-1`、`1e-2`、`1e-3`。不要只凭感觉选学习率。

5. 数据增强开关对比。

   关闭增强和开启随机裁剪 + 翻转，各跑一组，观察 train acc 和 val acc 的变化。

6. BatchNorm train/eval 检查。

   同一个 checkpoint，在 eval 模式和 train 模式下跑验证集，观察指标差异。正常评估必须用 eval。

7. checkpoint 恢复。

   训练一小段保存，再加载继续训练。确认 epoch、loss、acc 能连续。

## 8. 完成后应该能回答的问题

完成这一节后，你应该能回答：

- 为什么训练前要先做随机输入 forward？
- 为什么小样本过拟合是深度学习调试里的关键步骤？
- CIFAR-100 随机猜准确率大约是多少？
- softmax + cross entropy 的梯度为什么可以写成 `prob - one_hot`？
- 训练模式和验证模式对 BatchNorm 有什么影响？
- learning rate 太大和太小时，loss 曲线分别会是什么样？
- 为什么 checkpoint 不应该只保存模型权重？
- 数据增强为什么可能降低 train acc，却提高 val acc？
- 完整训练日志应该至少包含哪些信息？

这些问题回答清楚后，你就不仅是在“跑模型”，而是在建立可复现的训练流程。

## 9. 下一任务为什么自然出现

模型训练完之后，很容易有一种错觉：看见 accuracy 数字，就觉得实验结束了。

但研究不是只汇报最高分。研究要回答：

- 哪个设计真的有用？
- 为什么它有用？
- 换掉它会怎样？
- 结果是否稳定？
- 失败样例暴露了什么问题？

所以最后一个任务不是继续堆代码，而是写实验报告。你要用对照实验分析数据增强、BatchNorm、残差连接分别带来了什么影响。

这一步会把你从“能训练模型的人”，推向“能解释模型实验的人”。
