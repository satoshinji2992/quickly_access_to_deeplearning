# Sampling

语言模型输出下一个 token 的 logits。解码策略决定如何从这个分布选择 token，不会修改模型参数。

## Greedy 与 temperature

Greedy 每步选择最大 logit，结果确定，但可能较早陷入重复模式。Temperature 先缩放 logits：

$$
p_i=\operatorname{softmax}(z_i/T).
$$

`T < 1` 使分布更尖，`T > 1` 使分布更平。`T = 0` 不应直接代入除法，代码中通常把它当作 greedy。

## Top-k 与 top-p

- Top-k 只保留概率最高的 `k` 个候选，其余设为零后重新归一化。
- Top-p 先按概率降序排列，保留累计概率首次达到 `p` 的最短前缀。因此候选个数会随当前分布变化。

temperature、top-k 和 top-p 可以组合使用，但不存在适合所有任务的固定参数。测试时应固定随机种子，并检查被截断候选的概率确实为零。

可运行实现和边界检查见 [Task 29](../../block_03_transformer/task_29_generate_sampling/README.md)。
