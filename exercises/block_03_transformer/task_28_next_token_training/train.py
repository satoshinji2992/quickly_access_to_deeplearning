"""Offline next-token training on a small, coherent text corpus.

No download or third-party tokenizer is required.  The character tokenizer is
deliberately simple so the complete text -> ids -> padded batch -> masked loss
pipeline remains inspectable in one file.
"""

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, Dataset


CORE = Path(__file__).resolve().parents[1] / "task_27_minimind_core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from minimind_core import MiniMindConfig, MiniMindCore  # noqa: E402


DEFAULT_CORPUS = """
清晨，小城的雨停了。屋檐上的水滴慢慢落下，街边的树叶在风里发亮。
一个孩子背着书包走过石桥，他数着河面上的波纹，也数着远处的钟声。
老人打开面包店的木门，暖气带着麦香涌到街上。早起的人们互相问好，然后走向各自的一天。
中午，阳光穿过云层。孩子在课堂上写下：学习像搭桥，每一块石头都要放在正确的位置。
老师说，好问题比快答案更重要。先观察，再猜想，然后用实验检查猜想。
傍晚，河水映着灯光。孩子回到石桥，他发现早晨的问题已经有了新的答案。
他也明白，答案不是终点。当一个问题被说清楚，下一个问题就会从它身后走出来。
夜里又下起小雨，屋檐上的水滴慢慢落下。小城安静了，但新的故事正在开始。
""".strip()


class CharacterTokenizer:
    PAD = "<pad>"
    UNK = "<unk>"
    BOS = "<bos>"
    EOS = "<eos>"

    def __init__(self, token_to_id):
        self.token_to_id = dict(token_to_id)
        self.id_to_token = [None] * len(self.token_to_id)
        for token, index in self.token_to_id.items():
            self.id_to_token[index] = token
        required = (self.PAD, self.UNK, self.BOS, self.EOS)
        if any(token not in self.token_to_id for token in required):
            raise ValueError("tokenizer vocabulary is missing a special token")

    @classmethod
    def fit(cls, text):
        specials = [cls.PAD, cls.UNK, cls.BOS, cls.EOS]
        vocabulary = specials + sorted(set(text))
        return cls({token: index for index, token in enumerate(vocabulary)})

    @property
    def pad_token_id(self):
        return self.token_to_id[self.PAD]

    @property
    def bos_token_id(self):
        return self.token_to_id[self.BOS]

    @property
    def eos_token_id(self):
        return self.token_to_id[self.EOS]

    @property
    def vocab_size(self):
        return len(self.id_to_token)

    def encode(self, text, add_bos=False, add_eos=False):
        ids = [self.token_to_id.get(char, self.token_to_id[self.UNK]) for char in text]
        if add_bos:
            ids.insert(0, self.bos_token_id)
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids, skip_special_tokens=True):
        specials = {self.PAD, self.UNK, self.BOS, self.EOS}
        tokens = []
        for index in ids:
            token = self.id_to_token[int(index)]
            if skip_special_tokens and token in specials:
                continue
            tokens.append(token)
        return "".join(tokens)

    def state_dict(self):
        return {"token_to_id": self.token_to_id}

    @classmethod
    def from_state_dict(cls, state):
        return cls(state["token_to_id"])


class NextTokenDataset(Dataset):
    """Fixed-size blocks; neighbors share one token and the final block is padded."""

    def __init__(self, token_ids, seq_len, pad_token_id):
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if len(token_ids) < 2:
            raise ValueError("a split needs at least two tokens")
        self.ids = torch.tensor(token_ids, dtype=torch.long)
        self.seq_len = seq_len
        self.pad_token_id = pad_token_id
        self.starts = list(range(0, len(token_ids) - 1, seq_len))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        start = self.starts[index]
        block = self.ids[start : start + self.seq_len + 1]
        if block.numel() < self.seq_len + 1:
            padding = block.new_full((self.seq_len + 1 - block.numel(),), self.pad_token_id)
            block = torch.cat((block, padding))
        input_ids, labels = block[:-1], block[1:]
        attention_mask = input_ids.ne(self.pad_token_id)
        return input_ids, labels, attention_mask


def split_corpus(text, train_fraction=0.85):
    if not 0.5 <= train_fraction < 1.0:
        raise ValueError("train_fraction must be in [0.5, 1.0)")
    if len(text) < 40:
        raise ValueError("the corpus is too short for independent train/validation splits")
    split = int(len(text) * train_fraction)
    return text[:split], text[split:]


def make_dataloaders(text, seq_len=48, batch_size=8, seed=0):
    train_text, val_text = split_corpus(text)
    # Fitting only on the training split keeps validation genuinely held out.
    tokenizer = CharacterTokenizer.fit(train_text)
    train_ids = tokenizer.encode(train_text, add_bos=True, add_eos=True)
    val_ids = tokenizer.encode(val_text, add_bos=True, add_eos=True)
    train_data = NextTokenDataset(train_ids, seq_len, tokenizer.pad_token_id)
    val_data = NextTokenDataset(val_ids, seq_len, tokenizer.pad_token_id)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    return tokenizer, train_loader, val_loader


@torch.no_grad()
def evaluate(model, data_loader, device="cpu"):
    model.eval()
    loss_total, token_count = 0.0, 0
    for input_ids, labels, attention_mask in data_loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        attention_mask = attention_mask.to(device)
        _, loss = model(input_ids, labels, attention_mask=attention_mask)
        valid = attention_mask.bool() & labels.ne(-100)
        if model.config.pad_token_id is not None:
            valid = valid & labels.ne(model.config.pad_token_id)
        valid_count = int(valid.sum())
        # The model returns the mean over non-masked target tokens.  Aggregate
        # by that same denominator; weighting by batch size biases validation
        # whenever the last sequence contains more padding.
        loss_total += float(loss) * valid_count
        token_count += valid_count
    return loss_total / max(token_count, 1)


def save_checkpoint(path, model, optimizer, tokenizer, step, val_loss):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": asdict(model.config),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "tokenizer": tokenizer.state_dict(),
            "step": int(step),
            "val_loss": float(val_loss),
        },
        path,
    )


def load_checkpoint(path, device="cpu"):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch before the weights_only argument existed.
        checkpoint = torch.load(path, map_location=device)
    tokenizer = CharacterTokenizer.from_state_dict(checkpoint["tokenizer"])
    model = MiniMindCore(MiniMindConfig(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model, tokenizer, checkpoint


def train_model(
    text=DEFAULT_CORPUS,
    steps=80,
    seq_len=48,
    batch_size=8,
    device="cpu",
    seed=0,
    checkpoint_path=None,
):
    torch.manual_seed(seed)
    tokenizer, train_loader, val_loader = make_dataloaders(
        text, seq_len=seq_len, batch_size=batch_size, seed=seed
    )
    config = MiniMindConfig(
        vocab_size=tokenizer.vocab_size,
        dim=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        hidden_dim=128,
        max_seq_len=seq_len,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = MiniMindCore(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    iterator = iter(train_loader)
    model.train()
    for step in range(1, steps + 1):
        try:
            input_ids, labels, attention_mask = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            input_ids, labels, attention_mask = next(iterator)
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        attention_mask = attention_mask.to(device)
        _, loss = model(input_ids, labels, attention_mask=attention_mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % max(steps // 4, 1) == 0:
            print(f"step={step:03d} train_loss={loss.item():.4f}")

    val_loss = evaluate(model, val_loader, device)
    print(f"validation_loss={val_loss:.4f}")
    if checkpoint_path is not None:
        save_checkpoint(checkpoint_path, model, optimizer, tokenizer, steps, val_loss)
        print(f"checkpoint={Path(checkpoint_path)}")
    return model, tokenizer, val_loss


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", type=Path, help="optional UTF-8 corpus; built-in prose is used offline")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seq-len", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=Path("minimind_demo.pt"))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    text = args.text.read_text(encoding="utf-8") if args.text else DEFAULT_CORPUS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} corpus_chars={len(text)}")
    train_model(
        text=text,
        steps=args.steps,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed,
        checkpoint_path=args.checkpoint,
    )


if __name__ == "__main__":
    main()
