# Block 3：Transformer 与 MiniMind

第三块进入文本生成。

这里不再手写 NumPy 反向传播，而是使用 PyTorch。重点放在模块结构：attention、位置编码、RoPE、多头注意力、SwiGLU、RMSNorm、多层 decoder-only 模型，以及对齐 MiniMind 的小型训练流程。

## 任务安排

| 任务 | 问题 | 重点 |
| --- | --- | --- |
| task_20 | Transformer 在解决什么 | token、QKV、mask、decoder-only |
| task_21 | 交叉注意力怎么工作 | query 与 key/value 来自不同来源 |
| task_22 | 正弦位置编码怎么写 | sin/cos 编码与可视化 |
| task_23 | RoPE 为什么作用在 Q/K | 旋转位置编码 |
| task_24 | 多头注意力怎么实现 | causal mask、MHA、可选 GQA |
| task_25 | 前馈网络为什么不只是 Linear | FFN、SwiGLU |
| task_26 | token 怎么接进模型 | embedding、LM head、next-token loss |
| task_27 | 多层 Transformer 怎么堆 | Pre-Norm、残差连接 |
| task_28 | MiniMind Core 怎么搭 | RMSNorm、RoPE、SwiGLU、GQA |
| task_29 | 怎么训练具体任务 | pretrain、SFT、checkpoint、生成 |

## 通过标准

- 每个模块都有 shape test。
- causal mask 不允许偷看未来 token。
- 小语料可以 overfit。
- MiniMind Core 能完成一次小规模 pretrain 和 SFT smoke run。

