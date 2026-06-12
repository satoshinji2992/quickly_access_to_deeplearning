# Block 2 参考实现

这里放 NumPy ResNet 相关任务的参考实现。

当前文件：

- [resnet_numpy_solution.py](./resnet_numpy_solution.py)
- [train_cifar100_solution.py](./train_cifar100_solution.py)

运行前先在仓库根目录创建并激活 conda 环境:

```bash
conda env create -f environment.yml
conda activate deep-learning-intro
```

它包含：

- 图像数据管线里的 `to_nchw`、`normalize`、`iterate_minibatches`
- `im2col` / `col2im`
- `Conv2D`
- `MaxPool2D`
- `GlobalAvgPool2D`
- `BatchNorm2D`
- `BasicBlock`
- 一个很小的 `SmallResNet`

它会复用 `common/my_dl_lib.py` 里的通用组件:

- `Linear`
- `ReLU`
- `CrossEntropyLoss`
- `Momentum`
- `AdamW`

Block 2 自己只实现图像网络特有的卷积、池化、BatchNorm 和残差块.

运行训练:

```bash
python solutions/block_02_resnet/resnet_numpy_solution.py
```

这个脚本用的是合成小图片，不会下载 CIFAR-100。它展示了 ResNet 参考实现的一次完整 forward/backward/update 训练流程。

训练 CIFAR-100:

```bash
python solutions/block_02_resnet/train_cifar100_solution.py --epochs 5 --batch-size 32
```

第一次运行会用 `torchvision` 下载 CIFAR-100 到仓库根目录的 `data/`，每轮训练后会把参数、优化器状态和日志保存到 `checkpoints/cifar100_numpy_resnet.npz`。

建议先跑小样本过拟合，确认模型和反传能学住:

```bash
python solutions/block_02_resnet/train_cifar100_solution.py --overfit 200 --epochs 20 --batch-size 20 --channels 8 16 32 --lr 0.03
```

中断后继续训练:

```bash
python solutions/block_02_resnet/train_cifar100_solution.py --resume --epochs 10 --batch-size 32
```

标准 ResNet-18/50 本身通常不靠 Dropout，而是主要靠 BatchNorm、数据增强和 weight decay。这里的 NumPy 版本也按这个思路实现：卷积和全连接使用 He 初始化，残差块使用 BatchNorm，训练脚本提供随机裁剪、水平翻转、Momentum/AdamW 和 checkpoint。
