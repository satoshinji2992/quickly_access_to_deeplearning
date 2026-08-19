# task_28：Next-token 训练

模型结构接好以后，文本还要变成监督样本。这条数据路径同时处理训练集与验证集的隔离、标签错位、attention/loss 两处 PAD 屏蔽，以及 checkpoint 恢复后的 logits 一致性。

`train.py` 内置一段 UTF-8 中文短文，也接受 `--text` 指向本地文件。默认不联网，不依赖第三方 tokenizer，目的是让整条数据路径能在一个文件里读完。

![Next-token 标签、loss mask 与 causal/padding mask](assets/shifted_labels.png)

---

## 文本切分与词表

当前演示按字符位置切分：

```text
前 85% -> train text
后 15% -> validation text
```

两段不重叠。字符 tokenizer 只在 train text 上拟合，validation 中未见字符映射到 `<unk>`。四个特殊 token 是：

```text
<pad>  batch 补齐
<unk>  词表外字符
<bos>  序列开始
<eos>  序列结束
```

这里采用字符 tokenizer 是为了透明：一个字符对应一个 id。它没有 BPE、SentencePiece 的压缩效率，也不代表实际大规模语言模型的数据处理方式。

连续切分适合这个 smoke test，但不是严谨的语料评估方案。处理真实文档时通常还要按文档去重、避免同源文本跨 split，并单独保留 test set。

## Dataset 如何错开标签

`NextTokenDataset` 每次取 `seq_len+1` 个 id：

```text
block: [BOS, t0, t1, t2, EOS]
input: [BOS, t0, t1, t2]
label: [t0,  t1, t2, EOS]
```

起点按 `0, seq_len, 2*seq_len, ...` 前进。因为每块取 `seq_len+1` 个 id，相邻两块会共享一个边界 token。它在前一块是最后一个 label，在后一块是第一个 input，所以每条 next-token 转移只训练一次。

最后一块不足时补 PAD。这个方案没有高性能预训练常见的复杂 packing，但 input、label 和 padding 的对应关系容易手工检查。

## Padding 要走两条路径

Dataset 返回：

```text
input_ids:      (B,T)
labels:         (B,T)
attention_mask: (B,T), True 表示有效输入
```

模型用 attention mask 屏蔽 PAD key；loss 端则忽略 label 为 PAD 或 query mask 为 False 的位置。Causal mask 负责“不能看未来”，padding mask 负责“不能读取填充值”，两者不能互相替代。

图中 EOS 对应的输入位置仍然有效，因此 attention mask 为 1；它的下一个 target 已是 PAD，所以 loss mask 为 0。这正是代码中的行为。

## 训练循环

`make_dataloaders` 只打乱训练集，验证集保持固定顺序。训练 loader 的 generator 由 `seed` 初始化，同一环境下数据顺序可复现。

每一步执行：

```text
model.train()
forward -> masked cross-entropy
optimizer.zero_grad()
backward
clip_grad_norm_(1.0)
AdamW.step()
```

默认模型配置为：

```text
D=64, layers=2, Q heads=4, KV heads=2
SwiGLU hidden=128, max_seq_len=seq_len
```

训练对象就是 Task 27 的 `MiniMindCore`。第一层 `q_proj.weight.grad` 能直接反映 attention 路径是否参与训练：在非退化 batch 上，它会出现非零梯度。如果 loss 在变化而 attention 梯度为零，损失下降就不能归因于 Transformer 主干的学习。

## Validation loss 的分母

模型返回有效 target token 上的 mean loss。验证集最后一批往往含更多 PAD，因此不能简单按 batch 数平均。`evaluate` 使用：

```text
sum(batch_loss * valid_target_count) / sum(valid_target_count)
```

验证时调用 `model.eval()` 和 `torch.no_grad()`，数据来自独立的 `val_loader`。当前脚本只在训练结束后报告一次 validation loss，没有 early stopping 或 best-checkpoint 选择。这个数值主要用于检查验证通路，不能从几十步的小语料实验推断泛化能力。

## Checkpoint 能恢复什么

保存内容包括：

```text
config
model_state
optimizer_state
tokenizer
step
val_loss
```

`load_checkpoint` 用 config 重建模型，再载入参数和 tokenizer。一次完整 round-trip 呈现为：

```text
保存前 eval logits == 加载后 eval logits（浮点容差内）
weight tying 仍成立
tokenizer 映射一致
step / val_loss 元数据一致
```

CLI 暂未提供 `--resume`。虽然文件中保存了 `optimizer_state`，继续训练时仍需调用者创建 optimizer 并显式 `load_state_dict`；“能保存优化器”不等于“已实现完整断点续训”。

## 短实验

```bash
python exercises/block_03_transformer/task_28_next_token_training/train.py \
  --steps 4 --seq-len 24 --batch-size 2 \
  --checkpoint /tmp/minimind_smoke.pt
```

终端输出形如：

```text
device=... corpus_chars=...
step=001 train_loss=...
...
validation_loss=...
checkpoint=/tmp/minimind_smoke.pt
```

这段输出中的 loss 为有限数，最后会写出 checkpoint 文件。80 步的默认演示命令是：

```bash
python exercises/block_03_transformer/task_28_next_token_training/train.py \
  --steps 80 --checkpoint /tmp/minimind_demo.pt
```

换成本地文本：

```bash
python exercises/block_03_transformer/task_28_next_token_training/train.py \
  --text path/to/corpus.txt --steps 200 \
  --checkpoint /tmp/minimind_demo.pt
```

文本过短时脚本会报错；train/validation 两侧都需要至少两个 token。

同一组性质也收录在 Block 3 测试中：

```bash
python -m unittest discover -s tests -p 'test_block3.py' -v
```

测试覆盖 tokenizer 的训练集拟合范围、split 样本隔离、input/label 错位、causal/padding/loss 三类 mask、attention 非零梯度、固定小 batch 的过拟合现象、按有效 token 加权的 validation，以及 checkpoint round-trip 前后的 eval logits。

这条管线验证的是：

```text
text -> tokenizer -> dataset -> masks -> Transformer
     -> loss -> backward -> validation -> checkpoint
```

它没有大规模语料、成熟 tokenizer、分布式训练、系统评测或可用的对话能力，因此这里的结果代表微型训练管线闭环，不等同于从头训练通用 LLM。下一节转向 logits 到 token 的逐步生成过程。
