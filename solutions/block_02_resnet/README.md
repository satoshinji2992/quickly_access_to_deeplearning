# Block 2 参考实现

文件：

- [resnet_numpy_solution.py](./resnet_numpy_solution.py)
- [train_cifar100_solution.py](./train_cifar100_solution.py)

运行前先在仓库根目录创建并激活 conda 环境:

```bash
conda create -n dl_tutorial python=3.10
conda activate dl_tutorial
pip install -r requirements.txt
```

`resnet_numpy_solution.py` 包含：

- 图像数据管线里的 `to_nchw`、`normalize`、`iterate_minibatches`
- `im2col` / `col2im`
- `Conv2D`
- `MaxPool2D`
- `GlobalAvgPool2D`
- `BatchNorm2D`
- `BasicBlock`
- `SmallResNet`

它会复用 `common/my_dl_lib.py` 里的通用组件:

- `Linear`
- `ReLU`
- `CrossEntropyLoss`
- `Momentum`
- `AdamW`

运行训练:

```bash
python solutions/block_02_resnet/resnet_numpy_solution.py
```

这个脚本使用小规模图像数据，展示 ResNet 的完整 forward/backward/update 训练流程。

训练 CIFAR-100:

```bash
python solutions/block_02_resnet/train_cifar100_solution.py --epochs 5 --batch-size 32
```

第一次运行会用 `torchvision` 下载 CIFAR-100 到仓库根目录的 `data/`，每轮训练后会把参数、优化器状态和日志保存到 `checkpoints/cifar100_numpy_resnet.npz`。

小样本训练:

```bash
python solutions/block_02_resnet/train_cifar100_solution.py --overfit 200 --epochs 20 --batch-size 20 --channels 8 16 32 --lr 0.03
```

中断后继续训练:

```bash
python solutions/block_02_resnet/train_cifar100_solution.py --resume --epochs 10 --batch-size 32
```

这个 NumPy 版本使用 He 初始化、BatchNorm、随机裁剪、水平翻转、Momentum/AdamW 和 checkpoint。
