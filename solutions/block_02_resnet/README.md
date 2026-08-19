# Block 2 参考实现

本目录提供两个入口：

- [`resnet_numpy_solution.py`](./resnet_numpy_solution.py)：从 task 10～14 的已测试实现重新导出公共接口；
- [`train_cifar100_solution.py`](./train_cifar100_solution.py)：CIFAR-100 训练与完整 checkpoint。

参考模块不复制一份算子代码，避免主线实现修正后，solution 仍保留旧 bug。

---

## 环境

在仓库根目录运行：

```bash
conda create -n dl_tutorial python=3.10
conda activate dl_tutorial
pip install -r requirements.txt
```

CIFAR-100 由 `torchvision` 下载到 `data/`。纯 NumPy 卷积速度较慢，限制样本数和网络宽度可以显著缩短首次运行时间。

---

## 导出的接口

`resnet_numpy_solution.py` 包括：

```text
数据：to_nchw, normalize, stratified split/subset, minibatches
卷积：im2col, col2im, Conv2D
层：MaxPool2D, GlobalAvgPool2D, BatchNorm2D
模型：BasicBlock, SmallResNet
训练：one_hot, train_epoch, evaluate, augment_batch
```

通用的 `Linear`、`ReLU`、`CrossEntropyLoss` 和 `Momentum` 来自 `common/my_dl_lib.py`。

---

## 小样本运行

```bash
python solutions/block_02_resnet/train_cifar100_solution.py \
  --subset-size 200 --epochs 5 --batch-size 20 \
  --channels 8 16 32 --blocks 1 1 1 \
  --optimizer momentum --lr 0.03 --seed 0 --no-augment
```

`--subset-size` 是 `--train-limit` 的别名，二者不能同时传入。小样本拟合常关闭随机增强，所以示例显式传入 `--no-augment`；别名本身不会改变增强设置。

每轮输出：

```text
epoch=... train_loss=... train_acc=... val_loss=... val_acc=...
```

训练结束再输出 `test_loss/test_acc`。具体数值取决于配置和训练轮数，这里不承诺论文准确率。

不限制训练集的命令为：

```bash
python solutions/block_02_resnet/train_cifar100_solution.py \
  --epochs 5 --batch-size 32
```

这会使用默认 45,000/5,000/10,000 划分和 `(16,32,64)`、每 stage 两块的模型，NumPy 运行时间会很长。

---

## checkpoint 内容

默认文件：

```text
checkpoints/cifar100_numpy_resnet.npz
```

每轮结束调用 `save_checkpoint()`，保存：

```text
checkpoint_version
epoch
history / config
模型 train/eval 模式
全部命名参数
全部 BatchNorm running_mean / running_var
optimizer 类型、超参数、数组状态与 step
```

文件先写到同目录临时路径，再用 `replace()` 原子替换目标文件。

`load_checkpoint(..., strict=True)` 会检查模型键、数组 shape、optimizer 类型和 optimizer 状态。加载使用原位赋值，不会破坏 optimizer 保存的数组引用。

checkpoint round-trip 会比较：

- 新模型恢复后的 BN buffers 完全相同；
- eval logits 逐元素相同；
- optimizer velocity 和超参数相同；
- 原模型与恢复模型各继续一步后，参数仍完全相同。

---

## 恢复训练

若前一条小样本命令训练到第 5 轮，可继续到第 10 轮：

```bash
python solutions/block_02_resnet/train_cifar100_solution.py \
  --resume --epochs 10
```

输出先包含：

```text
restored model, optimizer, split, batch, seed, and augmentation config
resumed checkpoint=... start_epoch=6
```

随后训练第 6～10 轮。`--epochs` 表示目标总轮数，不是额外轮数。

恢复流程先读取 checkpoint config，再创建数据、模型和 optimizer。下列轨迹相关配置会自动恢复：

```text
train/val/test limit 与 val_size
batch size、seed 和增强选项
channels 与 blocks
optimizer 类型、learning rate 与 weight decay
```

`epochs` 是本次运行的目标总轮数，不从 checkpoint 覆盖；`data_dir` 可以指向新的数据位置。加载模型和 optimizer 数组时仍使用 strict 检查。

---

## 自动测试

```bash
python -m unittest discover -s tests -p 'test_block2.py' -v
```

正常结束时显示 `OK`。测试覆盖数据隔离、卷积梯度与参数更新、池化 tie rule、BatchNorm buffer、两种残差 shortcut、单 batch 拟合、epoch 指标加权和 checkpoint round-trip。
