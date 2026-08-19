# Tokenizer

语言模型接收 token id，不直接接收字符串。Tokenizer 定义了“文本如何切分”和“每个片段对应哪个 id”，并需保证 encode/decode 的特殊 token 规则一致。

```text
"我喜欢深度学习" -> tokenize -> token ids -> embedding
```

## 词表大小

词表较小时，文本往往被切成更多 token，序列和训练计算随之增长。词表较大时，embedding 与 LM head 的 `(V, D)` 参数量增长，低频条目也可能得不到充分训练。

## BPE 与 SentencePiece

BPE 反复合并训练语料中常见的相邻片段。SentencePiece 把分词建模为原始文本上的一个可复现步骤，不要求输入先按空格分词，因而也适用于中文等不依赖空格划分词的语言。

完整实验至少应记录：词表文件、正则化规则、BOS/EOS/PAD/UNK id、语料版本，以及 encode–decode round-trip 测试。本页尚未提供 tokenizer 训练代码。

原始方法可参考 [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) 与 [SentencePiece](https://arxiv.org/abs/1808.06226)。
