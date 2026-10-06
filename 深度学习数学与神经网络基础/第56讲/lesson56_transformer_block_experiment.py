# 第56讲实验：Transformer Block = Attention + Residual + Norm + MLP
# Python 3 + PyTorch

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(56)
torch.set_printoptions(precision=6, sci_mode=False)

class TinyPreNormBlock(nn.Module):
    def __init__(self, d_model=8, nhead=2, d_ff=16, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        a_in = self.norm1(x)
        a, _ = self.attn(a_in, a_in, a_in, attn_mask=attn_mask, need_weights=False)
        x = x + self.dropout(a)

        m_in = self.norm2(x)
        m = self.fc2(F.gelu(self.fc1(m_in)))
        x = x + self.dropout(m)
        return x

class TinyPostNormBlock(nn.Module):
    def __init__(self, d_model=8, nhead=2, d_ff=16, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        a, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = self.norm1(x + self.dropout(a))
        m = self.fc2(F.gelu(self.fc1(x)))
        x = self.norm2(x + self.dropout(m))
        return x

print("=== 实验1：Pre-Norm Block 的逐步 Shape ===")
B,T,D,H,DFF = 2,4,8,2,16
x = torch.randn(B,T,D)
block = TinyPreNormBlock(D,H,DFF,dropout=0.0)
mask = torch.zeros(T,T)
mask[torch.triu(torch.ones(T,T,dtype=torch.bool),diagonal=1)] = float("-inf")

with torch.no_grad():
    a_in = block.norm1(x)
    a, _ = block.attn(a_in,a_in,a_in,attn_mask=mask,need_weights=False)
    x1 = x + a
    m_in = block.norm2(x1)
    hidden = block.fc1(m_in)
    act = F.gelu(hidden)
    m = block.fc2(act)
    y_manual = x1 + m
    y_module = block(x,mask)

print("x.shape       =", tuple(x.shape))
print("norm1.shape   =", tuple(a_in.shape))
print("attn.shape    =", tuple(a.shape))
print("after add     =", tuple(x1.shape))
print("fc1.shape     =", tuple(hidden.shape))
print("gelu.shape    =", tuple(act.shape))
print("fc2.shape     =", tuple(m.shape))
print("block output  =", tuple(y_module.shape))
print("manual vs module max diff =", float(torch.max(torch.abs(y_manual-y_module))))

print("\n=== 实验2：MLP 是逐 Token 的，不做 Token mixing ===")
mlp = nn.Sequential(nn.Linear(D,DFF), nn.GELU(), nn.Linear(DFF,D))
x0 = torch.randn(1,4,D)
x1 = x0.clone()
x1[:,0,0] += 10.0
with torch.no_grad():
    m0 = mlp(x0)
    m1 = mlp(x1)
diff_per_token = torch.amax(torch.abs(m1-m0), dim=-1)
print("只改 Token1 后，各 Token 的 MLP 输出最大变化 =", diff_per_token.flatten().tolist())

print("\n=== 实验3：Causal Attention 可以把 Token1 的变化传到后续 Token ===")
attn = nn.MultiheadAttention(D,H,batch_first=True,bias=False)
with torch.no_grad():
    # 固定成可重复的普通随机权重即可
    torch.manual_seed(5601)
    for p in attn.parameters():
        p.copy_(torch.randn_like(p)*0.2)

a0,_ = attn(x0,x0,x0,attn_mask=mask,need_weights=False)
a1,_ = attn(x1,x1,x1,attn_mask=mask,need_weights=False)
attn_diff = torch.amax(torch.abs(a1-a0),dim=-1)
print("只改 Token1 后，各 Token 的 Attention 输出最大变化 =", attn_diff.flatten().tolist())
print("因果注意力下：未来 Token 可以受到更早 Token 的影响。")

print("\n=== 实验4：Residual Identity Path ===")
pre = TinyPreNormBlock(D,H,DFF,dropout=0.0)
post = TinyPostNormBlock(D,H,DFF,dropout=0.0)

def zero_sublayers(model):
    with torch.no_grad():
        for name,p in model.named_parameters():
            if "norm" not in name:
                p.zero_()

zero_sublayers(pre)
zero_sublayers(post)
with torch.no_grad():
    y_pre = pre(x,mask)
    y_post = post(x,mask)
print("Pre-Norm zero-sublayer: max |y-x| =", float(torch.max(torch.abs(y_pre-x))))
print("Post-Norm zero-sublayer: max |y-x| =", float(torch.max(torch.abs(y_post-x))))
print("Post-Norm 输出仍经过 LayerNorm，因此即使 F=0 也不等于原始 x。")

print("\n=== 实验5：Residual 对深层梯度传播的最小演示 ===")
torch.manual_seed(5602)
L = 30
dim = 32
alpha = 0.1
Ws = [torch.randn(dim,dim)*0.4 for _ in range(L)]

def forward_stack(inp, residual):
    h = inp
    for W in Ws:
        z = torch.tanh(h @ W)
        h = h + alpha*z if residual else alpha*z
    return h

xr = torch.randn(1,dim,requires_grad=True)
yn = forward_stack(xr, residual=False)
lossn = (yn**2).mean()
lossn.backward()
grad_no_res = xr.grad.detach().norm().item()

xr2 = xr.detach().clone().requires_grad_(True)
yr = forward_stack(xr2, residual=True)
lossr = (yr**2).mean()
lossr.backward()
grad_res = xr2.grad.detach().norm().item()

print("input grad norm without residual =", grad_no_res)
print("input grad norm with residual    =", grad_res)

print("\n=== 实验6：Pre-Norm 与 Post-Norm 的 Toy Gradient 对照 ===")
torch.manual_seed(5603)
pre2 = TinyPreNormBlock(D,H,DFF,dropout=0.0)
post2 = TinyPostNormBlock(D,H,DFF,dropout=0.0)

# 让注意力/MLP权重尽量一致，用于结构对照
with torch.no_grad():
    post2.attn.load_state_dict(copy.deepcopy(pre2.attn.state_dict()))
    post2.fc1.load_state_dict(copy.deepcopy(pre2.fc1.state_dict()))
    post2.fc2.load_state_dict(copy.deepcopy(pre2.fc2.state_dict()))
    post2.norm1.load_state_dict(copy.deepcopy(pre2.norm1.state_dict()))
    post2.norm2.load_state_dict(copy.deepcopy(pre2.norm2.state_dict()))

target = torch.randn(B,T,D)
xp = x.detach().clone().requires_grad_(True)
yp = pre2(xp,mask)
lp = F.mse_loss(yp,target)
lp.backward()
pre_grad = xp.grad.norm().item()

xq = x.detach().clone().requires_grad_(True)
yq = post2(xq,mask)
lq = F.mse_loss(yq,target)
lq.backward()
post_grad = xq.grad.norm().item()

print("toy pre-norm input grad norm  =", pre_grad)
print("toy post-norm input grad norm =", post_grad)
print("注意：单次 toy 数值不是一般定理，只用于观察结构差异。")

print("\n=== 实验7：Parameter Count ===")
D_big = 768
DFF_big = 3072
H_big = 12
mha = nn.MultiheadAttention(D_big,H_big,batch_first=True)
mlp = nn.Sequential(nn.Linear(D_big,DFF_big), nn.GELU(), nn.Linear(DFF_big,D_big))
ln1 = nn.LayerNorm(D_big)
ln2 = nn.LayerNorm(D_big)

def count(m): return sum(p.numel() for p in m.parameters())
print("MHA params =", count(mha))
print("MLP params =", count(mlp))
print("2x LayerNorm params =", count(ln1)+count(ln2))
print("Residual params = 0")
print("Total block-ish params =", count(mha)+count(mlp)+count(ln1)+count(ln2))

print("\n=== 实验8：train/eval 对 LayerNorm 与 Dropout 的影响不同 ===")
drop_block = TinyPreNormBlock(D,H,DFF,dropout=0.5)
drop_block.train()
torch.manual_seed(123)
yt1 = drop_block(x,mask)
torch.manual_seed(124)
yt2 = drop_block(x,mask)

drop_block.eval()
with torch.no_grad():
    ye1 = drop_block(x,mask)
    ye2 = drop_block(x,mask)
print("train mode two runs max diff =", float(torch.max(torch.abs(yt1-yt2))))
print("eval mode two runs max diff  =", float(torch.max(torch.abs(ye1-ye2))))
print("LayerNorm 本身 train/eval 行为一致；差异来自 Dropout。")
