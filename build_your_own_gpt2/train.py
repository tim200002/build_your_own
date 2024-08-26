import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters -------------------------------
B = 64 # batch size
BLOCK_SIZE = 256 # Token length, i.e. block size
LEARNING_RATE = 3e-4

NO_BLOCKS = 6
N_EMBD = 384
NO_HEADS = 6
device = "cuda" if torch.cuda.is_available() else "cpu"

max_iters = 5000
eval_iters = 500
eval_every_step = 200

torch.manual_seed(1337)

# Dataset ---------------------------------------
class Dataloader:
    def __init__(self, data_path, train_val_ratio):
        self.data_path = data_path
        self.train_val_ration = train_val_ratio

        with open("input.txt", "r") as f:
            text = f.read()
        
        self.alphabet = sorted(list(set(text)))
        self.vocab_size = len(self.alphabet)

        stoi = {c: i for i, c in enumerate(self.alphabet)}
        itos = {i: c for i, c in enumerate(self.alphabet)}
        self.encode = lambda x: [stoi[el] for el in x]
        self.decode = lambda x: "".join([itos[el] for el in x])

        data = torch.tensor(self.encode(text), dtype=torch.long) 

        n = int(train_val_ratio *len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

    def get_batch(self, split):
        data = self.train_data if split == "train" else self.val_data

        x_idx = torch.randint(len(data) - BLOCK_SIZE, (B,))
        x = torch.stack([data[x_id:x_id+BLOCK_SIZE] for x_id in x_idx])
        y = torch.stack([data[x_id +1 : x_id + 1 + BLOCK_SIZE] for x_id in x_idx])

        return x, y
    


# Model -----------------------------------------

class Head(nn.Module):
    def __init__(self, n_embd, head_size):
        super().__init__()

        self.head_size = head_size
        
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))) # BLOCK_SIZE, BLOCK_SIZE trial matrix

    def forward(self, x):
        # x: B, T, n_embd

        B, T, C = x.shape

        k = self.key(x) # B, T, head_size
        q = self.query(x) # B, T, head_size
        v = self.value(x) # B, T, head_size

        affin = k @ q.transpose(-2, -1) * (self.head_size)**(-0.5) # B, T, T
        affin_masked = torch.masked_fill(affin, self.tril[:T, :T] == 0, -torch.inf) 
        wei = F. softmax(affin_masked, dim=-1) # B, T, T

        out = wei @ v # B, T, head_size
        return out


class CausalMultiHeadAttention(nn.Module):
    def __init__(self, n_embd, no_heads, head_size):
        super().__init__()

        self.heads = nn.ModuleList([Head(n_embd, head_size) for _ in range(no_heads)])
        self.proj = nn.Linear(no_heads * head_size, n_embd)

    def forward(self, x):
        # x: B, T, n_embd
        out = torch.concat([head(x) for head in self.heads], dim=-1) # B, T, no_heads * head_size 
        out = self.proj(out)
        
        return out # B, BLOCK_SIZE, n_embd
    
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4* n_embd), # 4 is a HP
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd)
        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    def __init__(self, n_embd, no_heads):
        super().__init__()

        assert n_embd % no_heads == 0, "Must be divisable"
        head_size = n_embd // no_heads

        self.ffwd = FeedForward(n_embd)
        self.heads = CausalMultiHeadAttention(n_embd, no_heads, head_size)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # x: B, T, n_embd

        #? Is this weird double skip really intentionally
        x = x + self.heads(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
        

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head):
        super().__init__()
        self.token_embd_tab = nn.Embedding(vocab_size, n_embd)
        self.pos_embd_tab = nn.Embedding(BLOCK_SIZE, n_embd)

        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(NO_BLOCKS)])

        self.ln_f = nn.LayerNorm(n_embd)
        self.classifer = nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_embd = self.token_embd_tab(idx) # B, T, C
        pos_embd = self.pos_embd_tab(torch.arange(T, device=idx.device)) # B, T

        x = tok_embd + pos_embd # B, T, n_embd
        x = self.ln_f(self.blocks(x)) # B, T, n_embd
        logits = self.classifer(x) # B, T, vocab_size

        if targets is None:
            return x, None
        
        B, T, C = logits.shape
        logits_view = logits.view(-1, C)
        targets_view = targets.view(-1)

        loss = F.cross_entropy(logits_view, targets_view)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T), i.e. start initialization
        for _ in range(max_new_tokens):
            idx_window = idx[:, -BLOCK_SIZE:]
            logits, _ = self.forward(idx_window) # B, T, C
            logits = logits[:, -1, :] # B, C
            probs = F.softmax(logits, dim=-1) # B, C

            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat([idx, idx_next], dim=-1)

        return idx

# For eval
@torch.no_grad()
def estimate_loss(loader, eval_iters):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = loader.get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
   
# Training
loader = Dataloader("input.txt", 0.9)

model = GPTLanguageModel(loader.vocab_size, N_EMBD, NO_HEADS).to(device)
optim = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)


for iter in range(max_iters):
    optim.zero_grad(set_to_none=True)
    
    x, y = loader.get_batch("train")
    x, y = x.to(device), y.to(device)
    logits, loss = model(x, y)
    loss.backward()
    optim.step()

    if iter % eval_every_step == 0 or iter == max_iters -1:
        losses = estimate_loss(loader, 100)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")


# End by generating from the model
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(loader.decode(model.generate(context, max_new_tokens=500)[0].tolist()))





        
