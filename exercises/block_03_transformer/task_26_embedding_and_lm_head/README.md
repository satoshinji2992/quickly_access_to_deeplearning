# task_26：Embedding 与 LM Head

## 问题

token ID 不能直接进入模型，需要 embedding。模型输出也要映射回词表。

## 你要做什么

- 实现 token embedding。
- 实现 LM head。
- 实现 next-token 训练目标。

## 验收

- logits shape 为 `(batch, seq_len, vocab_size)`。
- loss 使用 shifted labels。
- 小语料能 overfit。

