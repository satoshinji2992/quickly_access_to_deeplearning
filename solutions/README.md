# 参考实现

这里放每个 block 的参考实现。

运行前先在仓库根目录创建并激活 conda 环境:

```bash
conda create -n dl_tutorial python=3.10
conda activate dl_tutorial
pip install -r requirements.txt
```

目录：

- `block_01_basics/linear_regression_solution.py`
- `block_01_basics/mini_network_reference.py`
- `block_02_resnet/resnet_numpy_solution.py`
- `block_02_resnet/train_cifar100_solution.py`
- `block_03_transformer/minimind_solution.py`

通用的 NumPy 小型深度学习库放在 `../common/my_dl_lib.py`; task_02 之后的 exercises 和 solutions 会复用它。
