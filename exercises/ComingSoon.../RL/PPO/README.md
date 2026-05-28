# PPO

PPO 是 RLHF 里常见的强化学习优化方法.

它的目标不是让模型从零学语言, 而是在已有模型基础上, 根据 reward model 的分数继续调整输出行为.

## 为什么要限制更新幅度?

如果每次更新太猛, 模型很容易偏离原来的语言分布.

结果可能是 reward 变高, 但回答变得奇怪.

PPO 会用 clipped objective 限制策略变化, 让模型别一步走太远.

## 在 LLM 里的位置

常见流程:

```text
pretrain -> SFT -> reward model -> PPO
```

PPO 不是唯一选择, 但它是理解 RLHF 的经典入口.
