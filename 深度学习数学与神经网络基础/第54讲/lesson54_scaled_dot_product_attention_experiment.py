# 第54讲实验：Scaled Dot-Product Attention
# Python 3 + NumPy + PyTorch

import math
import numpy as np
import torch
import torch.nn.functional as F

np.set_printoptions(precision=6, suppress=True)
torch.manual_seed(54)
np.random.seed(54)

def softmax_np(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

print("=== 实验1：完整手工 Attention（无 Mask）===")
Q = np.array([[1.,0.],[0.,1.],[1.,1.]], dtype=np.float64)
K = np.array([[1.,0.],[0.,1.],[1.,1.]], dtype=np.float64)
V = np.array([[1.,0.],[0.,2.],[3.,1.]], dtype=np.float64)
dk = Q.shape[-1]

scores = Q @ K.T
scaled = scores / math.sqrt(dk)
weights = softmax_np(scaled, axis=-1)
out = weights @ V

print("QK^T =\n", scores)
print("scaled scores =\n", scaled)
print("attention weights =\n", weights)
print("row sums =", weights.sum(axis=-1))
print("output =\n", out)

print("\n=== 实验2：Causal Mask ===")
T = Q.shape[0]
causal_mask = np.triu(np.ones((T,T), dtype=bool), k=1)
masked_scores = scaled.copy()
masked_scores[causal_mask] = -np.inf
masked_weights = softmax_np(masked_scores, axis=-1)
masked_out = masked_weights @ V
print("causal mask =\n", causal_mask.astype(int))
print("masked scores =\n", masked_scores)
print("masked weights =\n", masked_weights)
print("future weight max =", np.max(masked_weights[causal_mask]))
print("masked output =\n", masked_out)

print("\n=== 实验3：为什么除以 sqrt(d_k) ===")
dims = [8, 32, 128, 512]
num = 20000
for d in dims:
    q = np.random.randn(num, d)
    k = np.random.randn(num, d)
    dots = np.sum(q*k, axis=1)
    print(f"d={d:3d}  std(dot)={dots.std():.4f}  std(dot/sqrt(d))={(dots/math.sqrt(d)).std():.4f}")

print("\n=== 实验4：Softmax 饱和程度：缩放前 vs 缩放后 ===")
for d in [8,32,128,512]:
    q = np.random.randn(5000, d)
    keys = np.random.randn(5000, 16, d)
    raw = np.einsum("nd,nkd->nk", q, keys)
    p_raw = softmax_np(raw, axis=-1)
    p_scaled = softmax_np(raw / math.sqrt(d), axis=-1)
    max_raw = p_raw.max(axis=-1).mean()
    max_scaled = p_scaled.max(axis=-1).mean()
    ent_raw = (-p_raw*np.log(p_raw+1e-12)).sum(axis=-1).mean()
    ent_scaled = (-p_scaled*np.log(p_scaled+1e-12)).sum(axis=-1).mean()
    print(f"d={d:3d}  mean max prob raw={max_raw:.4f} scaled={max_scaled:.4f} | entropy raw={ent_raw:.4f} scaled={ent_scaled:.4f}")

print("\n=== 实验5：PyTorch scaled_dot_product_attention 对齐 ===")
Qt = torch.tensor(Q, dtype=torch.float32).view(1,1,3,2)
Kt = torch.tensor(K, dtype=torch.float32).view(1,1,3,2)
Vt = torch.tensor(V, dtype=torch.float32).view(1,1,3,2)
manual = torch.tensor(masked_out, dtype=torch.float32).view(1,1,3,2)
torch_out = F.scaled_dot_product_attention(Qt, Kt, Vt, is_causal=True, dropout_p=0.0)
print("manual output =\n", manual)
print("torch output  =\n", torch_out)
print("max abs diff =", float(torch.max(torch.abs(manual - torch_out))))

print("\n=== 实验6：输出是 Value 的加权和 ===")
w2 = masked_weights[1]
manual_token2 = w2[0]*V[0] + w2[1]*V[1] + w2[2]*V[2]
print("Token2 weights =", w2)
print("Token2 weighted sum =", manual_token2)
print("masked output token2 =", masked_out[1])
print("diff =", np.max(np.abs(manual_token2 - masked_out[1])))

print("\n=== 实验7：Mask 必须在 Softmax 前生效 ===")
# 错误示范：先 softmax，再把未来权重置 0，不重新归一化
wrong = weights.copy()
wrong[causal_mask] = 0.0
print("wrong row sums after zeroing future =", wrong.sum(axis=-1))
print("correct causal row sums =", masked_weights.sum(axis=-1))

print("\n=== 实验8：Backward：Q/K/V 都能收到梯度 ===")
Qb = torch.tensor(Q, dtype=torch.float32, requires_grad=True)
Kb = torch.tensor(K, dtype=torch.float32, requires_grad=True)
Vb = torch.tensor(V, dtype=torch.float32, requires_grad=True)
scores_b = Qb @ Kb.T / math.sqrt(dk)
mask_b = torch.triu(torch.ones(3,3,dtype=torch.bool), diagonal=1)
scores_b = scores_b.masked_fill(mask_b, float("-inf"))
attn_b = torch.softmax(scores_b, dim=-1)
out_b = attn_b @ Vb
loss = (out_b**2).mean()
loss.backward()
print("loss =", float(loss))
print("||grad Q|| =", float(Qb.grad.norm()))
print("||grad K|| =", float(Kb.grad.norm()))
print("||grad V|| =", float(Vb.grad.norm()))

print("\n=== 实验9：Batch / Head Shape ===")
B,H,T,d = 2,4,5,8
Qh = torch.randn(B,H,T,d)
Kh = torch.randn(B,H,T,d)
Vh = torch.randn(B,H,T,d)
Sh = Qh @ Kh.transpose(-2,-1)
Wh = torch.softmax(Sh/math.sqrt(d), dim=-1)
Oh = Wh @ Vh
print("Q shape =", tuple(Qh.shape))
print("score shape =", tuple(Sh.shape))
print("weight shape =", tuple(Wh.shape))
print("output shape =", tuple(Oh.shape))
