# DPO / RLHF

SFT 之后, 模型会更像助手.

但它不一定知道哪种回答更好.

偏好对齐就是让模型学习人类更喜欢哪种输出.

## RLHF

RLHF 通常包含几步:

1. 收集人类偏好数据.
2. 训练 reward model.
3. 用 PPO 之类的方法优化语言模型.

大概形式是:

```text
同一个 prompt
回答 A 比回答 B 更好
```

reward model 学会给回答打分, 再用强化学习让模型更倾向高分回答.

## DPO

DPO(Direct Preference Optimization)更直接.

它不显式训练 reward model, 而是直接用偏好对优化模型.

数据仍然是 chosen / rejected:

```text
prompt
chosen response
rejected response
```

DPO 的工程复杂度通常比 RLHF 低, 所以很多小模型对齐会先考虑它.

## 它们解决什么?

偏好对齐不是让模型变聪明的全部答案.

它更像是在调整模型的行为:

- 少输出明显不好的回答.
- 更符合人类偏好.
- 更稳定地遵循指令.

底层能力仍然很依赖预训练和数据质量.
