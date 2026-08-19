# PPO

PPO（Proximal Policy Optimization）用新旧策略的概率比率更新 policy，并通过 clipped objective 限制单次更新的幅度：

$$
L^{\mathrm{clip}}=\mathbb E\!\left[\min\left(r_tA_t,
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t\right)\right].
$$

在 RLHF 中，`A_t` 通常结合 reward model 分数、value 估计和相对参考模型的 KL 惩罚。Clipping 不是稳定训练的充分条件；reward 尺度、advantage 归一化、生成长度与采样参数仍需要记录和检查。

```text
pretrain -> SFT -> preference/reward data -> rollout -> PPO update
```

本页尚未包含 rollout worker、reward model、value head 或 PPO 训练器，因此不构成可运行 RLHF 教程。算法定义见 [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)。
