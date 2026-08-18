import torch
from torch import nn
from torch.nn import functional as F


class TinyLanguageModel(nn.Module):
    def __init__(self, vocab_size, dim, max_seq_len, pad_token_id=0):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(max_seq_len, dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        # Both modules reference the same Parameter, rather than copying its value.
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids, labels=None, attention_mask=None):
        b, t = input_ids.shape
        if t > self.max_seq_len:
            raise ValueError("sequence length exceeds max_seq_len")
        pos = torch.arange(t, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(pos)[None, :, :]
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            targets = labels.clone()
            targets[targets == self.pad_token_id] = -100
            if attention_mask is not None:
                targets[~attention_mask.bool()] = -100
            valid = targets.ne(-100)
            loss = (
                F.cross_entropy(logits[valid], targets[valid])
                if valid.any()
                else logits.sum() * 0.0
            )
        return logits, loss


if __name__ == "__main__":
    model = TinyLanguageModel(vocab_size=40, dim=16, max_seq_len=8)
    ids = torch.tensor([[1, 7, 3, 2]])
    logits, _ = model(ids)
    print("logits:", tuple(logits.shape))
    print("weights shared:", model.lm_head.weight is model.token_embedding.weight)
