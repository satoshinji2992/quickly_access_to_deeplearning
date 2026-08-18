# 大模型后训练中的强化学习

强化学习通常位于预训练和 SFT 之后，用奖励信号调整模型生成行为。它不替代语言建模阶段，也不保证模型的底层能力会随 reward 同步提高。

```text
prompt -> policy 生成 -> reward / verifier -> policy update
```

PPO 显式优化策略，并通常用 clipping 和相对参考模型的 KL 惩罚限制更新。GRPO 用同一 prompt 下多个样本的组内相对 reward 构造 advantage，避免单独训练与 policy 同规模的 value model。二者都需要处理 reward scale、KL、采样分布和数值稳定性。

可验证奖励只能证明预先定义的判据是否满足，不应与通用回答质量等同。本目录的 PPO/GRPO 页面仍是概念笔记，没有 rollout、reward 或训练代码。
