# SFT

SFT 是 Supervised Fine-Tuning, 监督微调.

预训练语言模型学会的是“接着写”.

SFT 想让它更像在“回答问题”.

## 数据长什么样?

常见格式:

```text
instruction: 请解释什么是梯度下降
response: 梯度下降是一种优化方法...
```

或者多轮对话:

```text
user: ...
assistant: ...
user: ...
assistant: ...
```

训练目标仍然是 next-token prediction.

区别在于, loss 通常只算 assistant 的回答部分, 不让模型去模仿 user.

## SFT 改变了什么?

它不会凭空给模型塞进大量新知识.

它主要改变模型的回答方式:

- 更愿意遵循指令.
- 更像聊天助手.
- 输出格式更稳定.
- 对任务描述更敏感.

预训练像学语言和知识, SFT 更像学怎么回答.
