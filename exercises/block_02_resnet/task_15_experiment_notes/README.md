# 读一遍已经跑完的实验

训练日志要放回运行条件里读。下面是一份实际记录：同一条命令、完整输出、能直接确认的事实，以及暂时还不能确认的解释。原始记录保存在 [`NOTES.md`](./NOTES.md)。

## 这次运行想知道什么

问题很小：默认教学配置能否在三轮内学会合成条纹数据，还是只能维持四分类的 25% 随机水平？

```text
数据          synthetic，四类 8×8 条纹图片
模型          channels=(8,16,32)，blocks=(1,1,1)
优化器        Momentum，lr=0.03，beta=0.9
训练轮数      3
seed          0
随机增强      关闭；合成路径不调用 CIFAR 增强
```

运行命令：

```bash
python exercises/block_02_resnet/task_14_numpy_resnet_train/train_resnet.py \
  --synthetic --epochs 3 --seed 0 --eval-test
```

这里加了 `--eval-test`，因为这份已完成的记录包含了合成数据的 held-out test 结果。平时调整配置只看 validation，不加这个参数。

本仓库当前环境得到：

```text
epoch=1 train_loss=1.5097 train_acc=0.306 val_loss=1.1607 val_acc=0.500
epoch=2 train_loss=0.9462 train_acc=0.708 val_loss=0.8540 val_acc=0.750
epoch=3 train_loss=0.6541 train_acc=0.750 val_loss=0.5983 val_acc=0.750
test_loss=0.6137 test_acc=0.750
```

## 日志能直接告诉我们什么

先不猜原因，日志本身已经说明：

train loss 从 1.5097 降到 0.6541，validation loss 也连续下降；validation 和 test accuracy 都是 0.75，高于四分类的 0.25 随机水平。

这些数足以说明 forward、backward 和参数更新已经连通，模型也读到了条纹规律。但这里只有四类人工设计的图案，它不能代表 NumPy SmallResNet 在 CIFAR-100 上的效果。

“第三轮准确率没有继续增加，模型已经收敛”也说得太早。准确率只记录类别是否猜对，loss 仍在下降；验证集只有 12 张图，一两个样本就会带来明显波动。更长的曲线或重复 seed，才能让这个判断更有把握。

把观察、当前解释和还缺的证据放在一张表里，会更容易看出结论能走到哪一步：

| 观察 | 当前解释 | 还缺什么 |
| --- | --- | --- |
| 三轮 train loss 持续下降 | 参数确实在更新 | 有限差分与单 batch 过拟合测试 |
| val acc 从 0.50 到 0.75 | 模型读到了未参与更新的条纹 | 换 seed 后是否仍明显高于 0.25 |
| 第 2、3 轮 val acc 相同 | 可能只是离散准确率暂时没变 | 更多 epoch、逐样本概率或混淆矩阵 |

## 换成 CIFAR-100 后，指标的含义变了

CIFAR-100 有 100 类，随机 top-1 准确率约为 1%。只取 500 张训练图片时，每类平均只有 5 张；小验证集中的一个样本也会明显改变准确率。因此 `1.2%` 和 `1.4%` 这类差别，在一次小样本运行中没有多少可解释性。

三个 split 的职责仍然不同：

```text
train       更新参数，也用于确认模型能否拟合
validation  比较学习率、增强与结构
test        配置确定后报告一次
```

若反复根据 test accuracy 改配置，test 就已经被当成 validation 使用了。数值仍然能算，但不再是独立的最终评估。

## 一张误分类图能说到哪里

![CIFAR-100 官方测试集上的八个真实误分类](assets/misclassified_examples.png)

这张图不是上面 NumPy SmallResNet 的结果。它使用 [chenyaofo/pytorch-cifar-models](https://github.com/chenyaofo/pytorch-cifar-models) 发布的 [CIFAR-100 ResNet-20 权重](https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar100_resnet20-23dac2f1.pt)。该权重标注的 top-1 准确率为 `68.83%`。

[`scripts/render_cifar100_errors.py`](../../../scripts/render_cifar100_errors.py) 会校验权重 SHA-256，在官方 10,000 张 test 图片上完整推理，再用固定 seed 选择八个错误样本。测试索引、真实标签、预测标签和权重摘要也写进了 PNG metadata。

图里有几组值得继续看：

| true | pred | 从单图能看到的线索 |
| --- | --- | --- |
| `snake` | `worm` | 两类都呈细长轮廓，32×32 下纹理很少 |
| `girl` | `boy` | 主体像素很少，类别边界本身也需要更多样本核对 |
| `mountain` | `whale` | 大片背景色可能盖过对象形状 |
| `clock` | `poppy` | 单张图看不出稳定原因 |

右栏只是下一步检查的方向，不能从一张图推出模型犯错的原因。比如 `mountain → whale` 可能与背景有关，也可能只是这张图的 logits 很接近。需要先汇总 confusion matrix、查看同类错误是否重复，再看置信度和原图。

## 这份记录留下了什么

它保留了复跑所需的命令、seed、模型配置和原始指标，也明确区分了 synthetic SmallResNet 与第三方 CIFAR-100 ResNet-20。几个月后再看到 `val_acc=0.75`，仍能知道它来自四分类条纹数据，而不是误以为 CIFAR-100 达到了 75%。

实验日志记录某一组数据和配置实际跑出了什么，shape、梯度与状态恢复等实现约定则交给自动测试：

```bash
python -m unittest discover -s tests -p 'test_block2.py' -v
```

参考：[CS231n：观察训练过程](https://cs231n.github.io/neural-networks-3/)、[CIFAR-100 官方说明](https://www.cs.toronto.edu/~kriz/cifar.html)。
