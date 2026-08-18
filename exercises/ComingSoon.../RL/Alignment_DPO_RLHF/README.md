# DPO 与 RLHF

偏好数据通常记录同一 prompt 下的成对比较：

```text
prompt
chosen response
rejected response
```

## RLHF

经典 RLHF 流程先用偏好对训练 reward model，再用 PPO 等策略优化方法提高预测 reward。训练中通常加入相对参考模型的 KL 约束，避免 policy 在优化有限 reward 时过度偏移。

```text
preference pairs -> reward model -> rollout -> PPO -> policy
```

## DPO

DPO 利用 reward 与最优策略之间的变量替换，直接使 chosen 回答相对 rejected 回答的对数概率比更大，同时以固定 reference policy 为基准。它不需要在微调期间单独拟合 reward model 或运行 on-policy rollout，但仍依赖偏好数据、reference model 和温度系数 `beta`。

偏好优化调整的是给定数据与判据下的生成行为，不能概括为所有能力的统一提升。评估时可留出独立的偏好验证集，并同时观察胜率、KL、生成长度和通用能力回归。

本页尚未提供可运行实现。算法定义见 [DPO 原论文](https://arxiv.org/abs/2305.18290)。
