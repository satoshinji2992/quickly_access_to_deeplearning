# 把 SmallResNet 跑起来

前几页分别实现了数据管线、卷积、BatchNorm 和残差块。[`train_resnet.py`](./train_resnet.py) 把它们接成一个完整训练循环。

纯 NumPy 的 `im2col` 很适合看清数组怎样流动，代价是训练完整 CIFAR-100 会比 PyTorch 慢很多。所以这里准备了两条路：四类合成条纹用来看 forward、backward 和参数更新是否真的有效，CIFAR-100 小样本则用来接上真实的下载、划分和增强流程。

## 同一个模型，两种数据规模

两条路共用同一套宽度和深度：

```python
channels = (8, 16, 32)
blocks_per_stage = (1, 1, 1)
```

`num_classes` 不是这套结构的固定常数，而是由数据路径决定：

| 数据 | 输入 | `num_classes` |
| --- | --- | ---: |
| 合成条纹 | `(N,3,8,8)` | 4 |
| CIFAR-100 | `(N,3,32,32)` | 100 |

`SmallResNet` 也允许传入更宽、更深的配置，不过本页的两次运行都保持 `(8,16,32)` 和 `(1,1,1)` 不变，免得结构变化和数据变化混在一起。CIFAR-100 路径的 shape 是：

| 位置 | 运算 | 输出 shape |
| --- | --- | --- |
| input | CIFAR-100 batch | `(N,3,32,32)` |
| stem | `Conv3x3 → BN → ReLU` | `(N,8,32,32)` |
| stage 1 | 1 个 identity BasicBlock | `(N,8,32,32)` |
| stage 2 | 1 个 projection BasicBlock | `(N,16,16,16)` |
| stage 3 | 1 个 projection BasicBlock | `(N,32,8,8)` |
| pool | Global Average Pool | `(N,32)` |
| fc | Linear | `(N,100)` |

stage 2 和 stage 3 的第一个卷积都使用 stride 2，两条 shortcut 同时投影到相同 shape，中间不再插入 MaxPool。合成数据的图只有 `8×8`，因而对应路线会变成：

```text
(N,3,8,8) -> (N,8,8,8) -> (N,16,4,4) -> (N,32,2,2)
            -> GlobalAvgPool -> (N,32) -> Linear -> (N,4)
```

这个模型借用了 ResNet 的 BasicBlock，但并非标准的 ResNet-18 或 ResNet-20：每个 stage 只有一个块，projection 使用 `1×1 Conv + BN`，卷积层还保留 bias。

## 一批图片怎样走完训练循环

`model.train()` 在进入一轮数据前调用一次，然后每个 minibatch 再走完前向、反向和更新：

```text
model.train()
for images, labels in minibatches:
    可选：随机裁剪与翻转
    logits = model.forward(images)
    loss = loss_fn.forward(logits, one_hot(labels))
    model.backward(loss_fn.backward())
    optimizer.step()
```

公共 `CrossEntropyLoss` 接收 one-hot target，所以训练代码会先把整数标签变成 `(N,num_classes)`。很多框架可以直接传整数标签，这只是本仓库的接口选择，不是交叉熵本身的限制。

例如三分类标签 `[2,0]` 会变成 `[[0,0,1],[1,0,0]]`。模型仍输出未归一化的 logits，`CrossEntropyLoss` 在内部计算 Softmax；如果先给 logits 做一次 Softmax 再传进去，相当于重复归一化，损失和梯度都会变。100 类等概率预测的交叉熵为 `-log(1/100)≈4.605`，可以用作读 CIFAR-100 初始日志的参照，但随机初始化模型的输出并不严格等概率。

各层在 backward 时会原位覆盖自己的梯度数组，这份实现因而不需要另外调用 `zero_grad()`。优化器默认使用 Momentum：

```python
Momentum(model.parameters(), lr=0.03, beta=0.9)
```

### 一轮 loss 为什么不能直接平均 batch loss

交叉熵返回当前 batch 的平均值。假设前两个 batch 各有 16 张，最后一个只有 2 张；若把三个 loss 等权平均，最后两张会被放大八倍。

代码按样本数累计：

$$
L_{epoch}=\frac{\sum_b |B_b|L_b}{\sum_b |B_b|}
$$

训练和评估都使用这一写法。

日志里的 `train_loss` 还混合了一轮训练过程中不同时刻的模型：第一批用刚开始的参数计算，后面的 batch 使用已经更新过的参数。`val_loss` 则是在这一轮结束后，用最终参数统一计算。训练端还可能做随机增强、使用当前 batch 的 BN 统计，验证端却使用固定图片和 running statistics，因此出现 `val_loss < train_loss` 并不矛盾。若要更直接地比较拟合与泛化，可以在同一轮结束后分别调用 `evaluate()` 计算未增强训练集和验证集，再比较两者。

## 先看四类条纹能不能学会

合成数据包含四类 `8×8` 图片，每一类由不同通道和位置的条纹表示。随机猜测的准确率是 25%，而条纹规律足够简单，当前模型跑过几轮后，loss 应该出现明显下降。

```bash
python exercises/block_02_resnet/task_14_numpy_resnet_train/train_resnet.py \
  --synthetic --epochs 3 --seed 0 --eval-test
```

在本仓库当前环境中的一次输出是：

```text
epoch=1 train_loss=1.5097 train_acc=0.306 val_loss=1.1607 val_acc=0.500
epoch=3 train_loss=0.6541 train_acc=0.750 val_loss=0.5983 val_acc=0.750
test_loss=0.6137 test_acc=0.750
```

末位数字可能随 NumPy 版本改变，不必逐字一致。这次运行里，训练 loss 从约 1.5 降到 1 以下，验证准确率也离开了 25% 的随机水平，说明条纹中的信号确实传到了参数。反过来，loss 完全不动或出现 `NaN` 时，问题多半在数据、梯度或参数更新，单纯增加 epoch 很少能补救。

自动测试里还有一个更强的检查：固定四张图片重复训练 60 步，loss 要降到初值的 25% 以下。这个结果用于确认模型能够过拟合一个小 batch，并不说明它有怎样的泛化表现。

## 再换成 CIFAR-100 小样本

第一次运行会通过 `torchvision` 下载数据到仓库根目录的 `data/`：

```bash
python exercises/block_02_resnet/task_14_numpy_resnet_train/train_resnet.py \
  --epochs 1 \
  --train-limit 500 --val-limit 500 \
  --seed 0
```

这条命令仍使用 `(8,16,32)` 和 `(1,1,1)`。500 张训练图片平均每类只有 5 张，一轮训练的准确率没有稳定的目标值。这次运行的意义是看下载、划分、增强和 100 类输出是否已经连在一起。

默认会开启 padding crop 与水平翻转。想看固定输入是否能被拟合时，可以关闭增强：

```bash
python exercises/block_02_resnet/task_14_numpy_resnet_train/train_resnet.py \
  --epochs 5 \
  --train-limit 500 --val-limit 500 \
  --no-augment --seed 0
```

训练 loss 没有下降时，可以先看有限差分是否通过，再确认 `optimizer.step()` 后卷积权重确实改变。梯度和更新都正常，再检查图片与标签是否使用同一组 shuffle 索引，以及 BatchNorm 是否处于 train 模式。

## 评估时模型哪里变了

`evaluate()` 先调用 `model.eval()`，然后按固定顺序遍历数据：

```text
BatchNorm 读取 running statistics
running buffers 不再更新
不做随机增强
不打乱 batch
```

下一轮 `train_epoch()` 会重新调用 `model.train()`。验证集来自官方 train 的独立分层切分，可以反复用来比较配置；官方 test 默认不计算。配置定下来后，在原命令末尾加 `--eval-test`，才会多输出一行 `test_loss` 和 `test_acc`。

## 参数和运行统计量如何归到同一个模型

`SmallResNet.parameters()` 为优化器提供参数与梯度，`named_parameters()` 给它们稳定名称，`named_buffers()` 则收集所有 BatchNorm 的运行均值和方差。保存时，`state_dict()` 会同时复制参数与 buffer；恢复时，`load_state_dict()` 写入原数组而不是换成新数组，这样先创建的优化器仍然指向有效参数。

<details>
<summary>选读：保存 checkpoint 后怎样继续训练</summary>

文件读写放在 [`solutions/block_02_resnet/train_cifar100_solution.py`](../../../solutions/block_02_resnet/train_cifar100_solution.py)：

```bash
python solutions/block_02_resnet/train_cifar100_solution.py \
  --subset-size 500 --epochs 5 --batch-size 20 \
  --channels 8 16 32 --blocks 1 1 1 --lr 0.03
```

checkpoint 包含模型参数、BatchNorm buffers、Momentum 的 velocity、当前 epoch、配置和历史指标。只保存 `model.parameters()` 会漏掉 BatchNorm 的 `running_mean/running_var`，新模型的评估输出就无法原样恢复。

默认文件是 `checkpoints/cifar100_numpy_resnet.npz`。继续到第 10 轮：

```bash
python solutions/block_02_resnet/train_cifar100_solution.py \
  --resume --epochs 10
```

程序先读取保存时的结构与数据配置，再恢复数组和优化器状态；这里的 `--epochs 10` 表示总轮数，而不是额外再跑 10 轮。shuffle 和增强使用 `seed+epoch` 作为随机种子，所以从第 6 轮恢复时，也会接上对应轮次的数据顺序。

这个参考入口也只在指定 `--eval-test` 后评估官方测试集。比如已经训练到第 10 轮、配置也确定了，可以运行 `--resume --epochs 10 --eval-test`：模型会加载现有 checkpoint，直接报告测试指标，不再更新参数。

</details>

数值梯度、单 batch 拟合和 checkpoint 往返都放在 Block 2 的测试里：

```bash
python -m unittest discover -s tests -p 'test_block2.py' -v
```

[下一页](../task_15_experiment_notes/README.md) 把刚才的合成数据运行整理成一份完整记录。

参考：[CS231n：训练与评估](https://cs231n.github.io/neural-networks-3/)。
