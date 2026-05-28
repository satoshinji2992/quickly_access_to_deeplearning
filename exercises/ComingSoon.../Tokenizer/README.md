# Tokenizer

模型不直接读文字.

它读 token id.

Tokenizer 负责把文本切成 token, 再映射成数字.

```text
"我喜欢深度学习" -> [1203, 456, 7789, 42]
```

## 词表大小

词表太小, 一句话会被切得很碎, 序列变长.

词表太大, embedding 和 LM head 会变大, 低频 token 也更难学.

所以词表大小是一个很实际的取舍.

## BPE

BPE(Byte Pair Encoding)从字符或字节开始, 不断合并最常见的相邻片段.

它能把常见词切得短一点, 少见词也能拆开表示.

## SentencePiece

SentencePiece 常见于很多开源 LLM.

它可以直接从原始文本训练 tokenizer, 不强依赖空格分词. 对中文、日文这类语言很方便.

## 为什么 tokenizer 很重要?

同一句话, tokenizer 不同, token 数可能差很多.

token 数影响:

- 训练速度.
- 上下文长度.
- 生成成本.
- 模型对不同语言的友好程度.

所以 tokenizer 不是预处理小工具, 它会直接影响模型体验.
