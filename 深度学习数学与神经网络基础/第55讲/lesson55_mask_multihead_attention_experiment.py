# 第55讲实验：Mask 与 Multi-Head Attention
# Python 3 + PyTorch + NumPy

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(55)
np.random.seed(55)
torch.set_printoptions(precision=6, sci_mode=False)

print("=== 实验1：从 (B,T,D) 拆成多个 Head ===")
B,T,D,H = 1,3,4,2
d = D // H

X = torch.tensor([[
    [1.,0.,0.,1.],
    [0.,1.,0.,1.],
    [1.,1.,1.,0.],
]])

# 为了看清 head 行为，先令 Q=K=V=X，相当于 identity projection。
Q = X.clone()
K = X.clone()
V = X.clone()

Qh = Q.reshape(B,T,H,d).transpose(1,2)
Kh = K.reshape(B,T,H,d).transpose(1,2)
Vh = V.reshape(B,T,H,d).transpose(1,2)

print("X.shape =", tuple(X.shape))
print("after reshape =", (B,T,H,d))
print("after transpose Qh.shape =", tuple(Qh.shape))
print("Head1 Q =\n", Qh[0,0])
print("Head2 Q =\n", Qh[0,1])

print("\n=== 实验2：两个 Head 产生不同 Attention Pattern ===")
scores = Qh @ Kh.transpose(-2,-1) / math.sqrt(d)
mask = torch.triu(torch.ones(T,T,dtype=torch.bool), diagonal=1)
scores_masked = scores.masked_fill(mask, float("-inf"))
weights = torch.softmax(scores_masked, dim=-1)
head_out = weights @ Vh

print("scores.shape =", tuple(scores.shape))
print("weights.shape =", tuple(weights.shape))
print("Head1 weights =\n", weights[0,0])
print("Head2 weights =\n", weights[0,1])
print("Head1 output =\n", head_out[0,0])
print("Head2 output =\n", head_out[0,1])

print("\n=== 实验3：concat 多个 Head ===")
# 先把轴从 (B,H,T,d) 换回 (B,T,H,d)，再拼回 D。
concat = head_out.transpose(1,2).contiguous().reshape(B,T,D)
print("head_out.shape =", tuple(head_out.shape))
print("concat.shape =", tuple(concat.shape))
print("concat =\n", concat)

print("\n=== 实验4：Output Projection W_O ===")
WO = torch.tensor([
    [1.,0.,0.,1.],
    [0.,1.,1.,0.],
    [1.,1.,0.,0.],
    [0.,0.,1.,1.],
])
out = concat @ WO
print("W_O.shape =", tuple(WO.shape))
print("final output.shape =", tuple(out.shape))
print("final output =\n", out)

print("\n=== 实验5：手写结果与 F.scaled_dot_product_attention 对齐 ===")
sdpa = F.scaled_dot_product_attention(Qh,Kh,Vh,is_causal=True,dropout_p=0.0)
print("manual vs sdpa max diff =", float(torch.max(torch.abs(head_out - sdpa))))

print("\n=== 实验6：nn.MultiheadAttention 与手写结果对齐 ===")
mha = nn.MultiheadAttention(embed_dim=D, num_heads=H, bias=True, batch_first=True)
with torch.no_grad():
    # in_proj_weight = [W_Q; W_K; W_V]，这里都设为 identity。
    mha.in_proj_weight.zero_()
    mha.in_proj_weight[:D].copy_(torch.eye(D))
    mha.in_proj_weight[D:2*D].copy_(torch.eye(D))
    mha.in_proj_weight[2*D:].copy_(torch.eye(D))
    mha.in_proj_bias.zero_()
    # PyTorch Linear 保存 (out,in)，我们的 WO 数学写法 out = concat @ WO，
    # 因此 out_proj.weight = WO.T。
    mha.out_proj.weight.copy_(WO.T)
    mha.out_proj.bias.zero_()

float_mask = torch.zeros(T,T)
float_mask[torch.triu(torch.ones(T,T,dtype=torch.bool),diagonal=1)] = float("-inf")
mha_out, mha_w = mha(X,X,X,attn_mask=float_mask,need_weights=True,average_attn_weights=False)

print("mha_out.shape =", tuple(mha_out.shape))
print("mha weights shape =", tuple(mha_w.shape))
print("manual vs nn.MultiheadAttention max diff =", float(torch.max(torch.abs(out - mha_out))))

print("\n=== 实验7：Padding Mask 与 Causal Mask 可以组合 ===")
B2,H2,T2,d2 = 2,2,4,3
Q2 = torch.randn(B2,H2,T2,d2)
K2 = torch.randn(B2,H2,T2,d2)
V2 = torch.randn(B2,H2,T2,d2)
S2 = Q2 @ K2.transpose(-2,-1) / math.sqrt(d2)

# causal: (T,T)
causal = torch.triu(torch.ones(T2,T2,dtype=torch.bool), diagonal=1)
# padding: 第二条样本的最后一个 key 是 PAD。Shape (B,1,1,T)
padding = torch.zeros(B2,1,1,T2,dtype=torch.bool)
padding[1,0,0,-1] = True

combined = causal.view(1,1,T2,T2) | padding
S2m = S2.masked_fill(combined, float("-inf"))
W2 = torch.softmax(S2m, dim=-1)

print("scores shape =", tuple(S2.shape))
print("causal shape =", tuple(causal.shape))
print("padding shape =", tuple(padding.shape))
print("combined broadcast shape =", tuple(combined.shape))
print("sample2 PAD key max weight =", float(W2[1,:,:, -1].max()))

print("\n=== 实验8：Head Count 与参数量 ===")
for Dm,Hm in [(64,1),(64,2),(64,4),(64,8)]:
    layer = nn.MultiheadAttention(Dm,Hm,batch_first=True)
    params = sum(p.numel() for p in layer.parameters())
    print(f"D={Dm:3d}, H={Hm:2d}, d_head={Dm//Hm:2d}, params={params}")

print("\n=== 实验9：More heads 不等于更多 QKV/O 参数（D 固定）===")
Dm = 768
expected = 4*Dm*Dm + 4*Dm  # in_proj weight/bias + out_proj weight/bias
print("D=768 standard PyTorch MHA params =", expected)
for Hm in [1,3,6,12,24]:
    layer = nn.MultiheadAttention(Dm,Hm,batch_first=True)
    actual = sum(p.numel() for p in layer.parameters())
    print(f"H={Hm:2d}, d_head={Dm//Hm:3d}, params={actual}")

print("\n=== 实验10：改变一个 Head 的输出，只通过 concat + W_O 影响最终输出 ===")
head_out_changed = head_out.clone()
head_out_changed[:,0] += 1.0  # 只改 Head1
concat_changed = head_out_changed.transpose(1,2).contiguous().reshape(B,T,D)
out_changed = concat_changed @ WO
print("Head2 unchanged max diff =", float(torch.max(torch.abs(head_out_changed[:,1] - head_out[:,1]))))
print("final output max diff after changing Head1 =", float(torch.max(torch.abs(out_changed - out))))
