# 一次 SmallResNet 合成数据运行记录

记录时间：2026-09-21

## 问题

默认教学配置能否在三轮内学会四类条纹数据，并明显超过 25% 的随机准确率？

这次没有比较两个超参数，只确认数据、forward、backward 和参数更新能否学到一个简单规律。

## 环境与命令

```text
Python 3.13.7
NumPy 2.3.2
arm64
seed=0
```

```bash
python exercises/block_02_resnet/task_14_numpy_resnet_train/train_resnet.py \
  --synthetic --epochs 3 --seed 0 --eval-test
```

## 数据与模型

| 项目 | 本次取值 |
| --- | --- |
| 数据 | synthetic，四类 `8×8` 条纹图 |
| split | 72 train / 12 validation / 12 test |
| 随机增强 | 无 |
| channels | `(8,16,32)` |
| blocks | `(1,1,1)` |
| batch size | 16 |
| optimizer | Momentum，`beta=0.9` |
| learning rate | 0.03 |
| test evaluation | `--eval-test` |
| checkpoint | 未使用 |

## 原始输出

```text
epoch=1 train_loss=1.5097 train_acc=0.306 val_loss=1.1607 val_acc=0.500
epoch=2 train_loss=0.9462 train_acc=0.708 val_loss=0.8540 val_acc=0.750
epoch=3 train_loss=0.6541 train_acc=0.750 val_loss=0.5983 val_acc=0.750
test_loss=0.6137 test_acc=0.750
```

## 观察

- train loss 三轮连续下降，从 1.5097 到 0.6541。
- validation loss 同样连续下降。
- validation 与 test accuracy 都达到 0.75，高于 0.25 随机水平。
- 输出中没有 `NaN` 或 `Inf`。

## 当前解释

模型能从未参与更新的样本中识别条纹规律，说明数据、forward、backward 和优化器已经连通。这个结果只对应四类合成图片，不能外推成 CIFAR-100 准确率。

第 2、3 轮 validation accuracy 相同，也不足以说明模型已经收敛。验证集只有 12 张，猜对或猜错一张就会改变约 8.3 个百分点，而 loss 此时仍在下降。

## 与实现测试的关系

这次运行观察整体行为。更细的实现问题由 `tests/test_block2.py` 检查，包括有限差分、卷积参数是否更新、固定小 batch 是否能被过拟合，以及 checkpoint 恢复后 logits 是否一致。
