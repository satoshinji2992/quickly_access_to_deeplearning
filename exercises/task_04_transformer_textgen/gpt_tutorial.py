import torch
import torch.nn as nn
from torch.nn import functional as F
import random
import textwrap #文本换行

#超参数
batch_size = 32
block_size = 256 
n_embd = 384    
n_head = 6
n_layer = 3
head_size = n_embd // n_head

device = "cuda" if torch.cuda.is_available() else "cpu"
# 设置默认的数据类型
torch.set_default_dtype(torch.float32)  # 或 torch.float64 等

# 设置默认的设备
if torch.cuda.is_available():
    torch.set_default_device('cuda')
else:
    torch.set_default_device('cpu')

wrap_width = 50
learning_rate = 0.0001
max_iter = 2000
eval_iters = 200 
dropout_value = 0.1
torch.manual_seed(1337)
file_name = "./asset/Hong_Lou_Meng.txt"

with open(file_name, 'r', encoding = 'utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda str1: [stoi[c] for c in str1]
decode = lambda list1: ''.join([itos[i] for i in list1])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

print(f"文件{file_name}读取完成") 


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

#--------损失评测------------
@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            logits, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out    

#-------Head类--------------
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout_value)  
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, x):
        B, T, C = x.shape
        wei = self.key(x) @ self.query(x).transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, head_size, num_heads):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout_value)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),  # 或者使用 nn.ReLU()
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout_value)
        )
    
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.ln1 = nn.LayerNorm(n_embd)
        self.attention = MultiHeadAttention(head_size, n_head)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd)
    
    def forward(self, x):
        # 注意力块 + 残差连接
        x = x + self.attention(self.ln1(x))
        # 前馈网络 + 残差连接
        x = x + self.ffwd(self.ln2(x))
        return x

#---------------------
class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 嵌入层
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        # Transformer块
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(n_embd, n_head) for _ in range(n_layer)])
        
        # 最终层归一化
        self.ln_f = nn.LayerNorm(n_embd)
        
        # 语言模型头部
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
        # 初始化参数
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape #B=batch_size, T=block_size
        
        # 获取词嵌入和位置嵌入
        token_embd = self.token_embedding_table(idx)
        position_idx = torch.arange(T, device=idx.device)
        position_embd = self.position_embedding_table(position_idx)
        
        # 嵌入相加
        x = token_embd + position_embd
        
        # 通过Transformer块
        x = self.transformer_blocks(x)
        
        # 最终层归一化
        x = self.ln_f(x)
        
        # 预测下一个token
        logits = self.lm_head(x)
        
        # 计算损失
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, token_sequ, max_new_tokens):
        for _ in range(max_new_tokens):
            # 获取最后block_size个token
            token_input = token_sequ[:, -block_size:]
            
            # 获取预测
            logits, _ = self.forward(token_input)
            
            # 只关注最后一个位置的预测
            logits = logits[:, -1, :]
            
            # 获取概率分布
            probs = F.softmax(logits, dim=-1)
            
            # 采样下一个token
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # 将新token追加到序列中
            token_sequ = torch.cat((token_sequ, idx_next), dim=1)
            
        # 返回新生成的token
        new_tokens = token_sequ[:, -max_new_tokens:]
        return new_tokens
    
#---------运行-------------
def main():
    print(f"开始训练{file_name}...")
    model = LanguageModel()
    model = model.to(device)
    print(sum(p.numel() for p in model.parameters())/1e6, "M parameters")  

    # 设置优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # 设置学习率调度器（可选）
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iter)

    # 训练循环
    for i in range(max_iter):
        # 定期评估
        if i % eval_iters == 0:
            losses = estimate_loss(model)
            print(f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # 获取一批数据
        xb, yb = get_batch('train')
        
        # 前向传播
        logits, loss = model(xb, yb)
        
        # 反向传播
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        # 更新学习率（如果使用调度器）
        # scheduler.step()

    print("训练完成")
    
    # 生成文本
    max_new_tokens = 500
    start_idx = random.randint(0, len(val_data)-block_size-max_new_tokens)

    # 上文内容
    context = torch.zeros((1, block_size), dtype=torch.long, device=device)
    context[0, :] = val_data[start_idx:start_idx+block_size]
    context_str = decode(context[0].tolist())
    wrapped_context_str = textwrap.fill(context_str, wrap_width)

    # 真实下文
    real_next_tokens = torch.zeros((1, max_new_tokens), dtype=torch.long, device=device)
    real_next_tokens[0, :] = val_data[start_idx+block_size:start_idx+block_size+max_new_tokens]
    real_next_tokens_str = decode(real_next_tokens[0].tolist())
    wrapped_real_next_tokens_str = textwrap.fill(real_next_tokens_str, wrap_width)

    # 生成下文
    generated_next_tokens = model.generate(context, max_new_tokens)
    generated_next_tokens_str = decode(generated_next_tokens[0].tolist())
    wrapped_generated_next_tokens_str = textwrap.fill(generated_next_tokens_str, wrap_width)

    # 打印结果
    print("上文内容：")
    print(wrapped_context_str)
    print("---------------------------------------")
    print("真实下文：")
    print(wrapped_real_next_tokens_str)
    print("---------------------------------------")
    print("生成下文：")
    print(wrapped_generated_next_tokens_str)

if __name__ == "__main__":
    main()