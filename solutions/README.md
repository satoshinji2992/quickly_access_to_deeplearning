# 参考实现

这里放每个 block 的参考实现。

运行前先在仓库根目录创建并激活 conda 环境:

```bash
conda env create -f environment.yml
conda activate deep-learning-intro
```

使用建议：

1. 先完成 `exercises/` 中的 starter。
2. 跑不通时先看报错和 README。
3. 最后再看这里的参考实现。

当前已迁移：

- `block_01_basics/linear_regression_solution.py`
- `block_01_basics/mini_network_reference.py`
- `block_02_resnet/resnet_numpy_solution.py`
- `block_03_transformer/minimind_solution.py`

这里的参考实现不是唯一答案。它们主要用来对照 shape、forward/backward 写法和训练闭环。建议先做 `exercises/`，卡住后再回来翻。

通用的 NumPy 小型深度学习库放在 `../common/my_dl_lib.py`; task_02 之后的 exercises 和 solutions 会复用它。
