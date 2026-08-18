# Block 1 参考实现

这里包含两份可以单独运行的 NumPy 参考代码：

- `linear_regression_solution.py`：在一组手写点上实现 $y=ax+b$、MSE 和 full-batch 梯度下降；
- `mini_network_reference.py`：实现 2-4-4-2 分类网络，可切换圆形、方形、三角形数据及四种优化方法。

运行：

```bash
python solutions/block_01_basics/linear_regression_solution.py
python solutions/block_01_basics/mini_network_reference.py
```

参考代码用于核对思路，不与 exercise 保持逐行一致：

- task 00 的 `starter.py` 读取 `Salary_Data.csv`，参考实现使用内置小数组；
- task 01 的主线代码单独维护训练集与验证集，`mini_network_reference.py` 只演示网络计算，在同一批合成数据上训练和画图；
- task 02 的完整组件实现位于仓库根目录的 `common/my_dl_lib.py`。

对照时，shape、损失定义和梯度方向比最终数值更有辨别力。随机初始化、数据顺序和优化器状态都会影响日志，因此两份输出并不会逐项相等。
