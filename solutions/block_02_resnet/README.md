# Block 2 参考实现

这里放 NumPy ResNet 相关任务的参考实现。

当前文件：

- [resnet_numpy_solution.py](./resnet_numpy_solution.py)

它包含：

- 图像数据管线里的 `to_nchw`、`normalize`、`iterate_minibatches`
- `im2col` / `col2im`
- `Conv2D`
- `MaxPool2D`
- `GlobalAvgPool2D`
- `BatchNorm2D`
- `BasicBlock`
- 一个很小的 `SmallResNet`

运行 smoke test：

```bash
python solutions/block_02_resnet/resnet_numpy_solution.py
```

这个脚本用的是合成小图片，不会下载 CIFAR-100。它的作用是检查参考实现能不能完成一次 forward/backward/update。
