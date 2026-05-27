import torch
from torch import nn
from torch.nn import functional as F


class TinyLanguageModel(nn.Module):
    def __init__(self, vocab_size, dim, max_seq_len):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(max_seq_len, dim)
        self.lm_head = nn.Linear(dim, vocab_size)

    def forward(self, input_ids, labels=None):
        b, t = input_ids.shape
        pos = torch.arange(t, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(pos)[None, :, :]
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        return logits, loss

