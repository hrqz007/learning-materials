# 第53讲实验：Q/K/V Linear Projection 的数值、Shape、融合实现与梯度
# 运行环境：Python 3 + NumPy + PyTorch

import numpy as np
import torch
import torch.nn as nn

# 固定随机种子，保证结果可复现。
np.random.seed(53)
torch.manual_seed(53)

# 为了让手算清晰，使用 3 个 Token，每个 Token 有 4 个特征。
X_np = np.array([
    [1., 0., 1., 0.],
    [0., 1., 0., 1.],
    [1., 1., 0., 0.],
], dtype=np.float32)

# 分别定义 Q、K、V 的三套投影矩阵。
# 这里 D_model=4，d_q=d_k=d_v=3，因此矩阵 Shape 都是 (4, 3)。
WQ_np = np.array([
    [1., 0., 1.],
    [0., 1., 1.],
    [1., 1., 0.],
    [0., 1., -1.],
], dtype=np.float32)

WK_np = np.array([
    [1., 1., 0.],
    [1., 0., 1.],
    [0., 1., 1.],
    [1., -1., 0.],
], dtype=np.float32)

WV_np = np.array([
    [1., 0., 0.],
    [0., 2., 0.],
    [0., 0., 3.],
    [1., 1., 1.],
], dtype=np.float32)

print("=== 实验1：NumPy 直接计算 Q/K/V ===")
Q_np = X_np @ WQ_np
K_np = X_np @ WK_np
V_np = X_np @ WV_np
print("X.shape =", X_np.shape)
print("Q.shape =", Q_np.shape)
print("K.shape =", K_np.shape)
print("V.shape =", V_np.shape)
print("Q =\n", Q_np)
print("K =\n", K_np)
print("V =\n", V_np)

print("\n=== 实验2：手算第2个 Token 的 Query ===")
x2 = X_np[1]
q2_manual = np.array([
    x2[0]*WQ_np[0,0] + x2[1]*WQ_np[1,0] + x2[2]*WQ_np[2,0] + x2[3]*WQ_np[3,0],
    x2[0]*WQ_np[0,1] + x2[1]*WQ_np[1,1] + x2[2]*WQ_np[2,1] + x2[3]*WQ_np[3,1],
    x2[0]*WQ_np[0,2] + x2[1]*WQ_np[1,2] + x2[2]*WQ_np[2,2] + x2[3]*WQ_np[3,2],
], dtype=np.float32)
print("x2 =", x2)
print("手算 q2 =", q2_manual)
print("矩阵乘法 q2 =", Q_np[1])
print("最大误差 =", np.max(np.abs(q2_manual - Q_np[1])))

print("\n=== 实验3：PyTorch nn.Linear 与 NumPy 对齐 ===")
X = torch.tensor(X_np)
q_proj = nn.Linear(4, 3, bias=False)
k_proj = nn.Linear(4, 3, bias=False)
v_proj = nn.Linear(4, 3, bias=False)

# PyTorch Linear 保存 weight.shape=(out_features, in_features)，
# 因此要把 NumPy 的 (in, out) 权重转置后复制进去。
with torch.no_grad():
    q_proj.weight.copy_(torch.tensor(WQ_np.T))
    k_proj.weight.copy_(torch.tensor(WK_np.T))
    v_proj.weight.copy_(torch.tensor(WV_np.T))

Q_t = q_proj(X)
K_t = k_proj(X)
V_t = v_proj(X)
print("q_proj.weight.shape =", tuple(q_proj.weight.shape))
print("NumPy vs PyTorch Q 最大误差 =", float(np.max(np.abs(Q_np - Q_t.detach().numpy()))))
print("NumPy vs PyTorch K 最大误差 =", float(np.max(np.abs(K_np - K_t.detach().numpy()))))
print("NumPy vs PyTorch V 最大误差 =", float(np.max(np.abs(V_np - V_t.detach().numpy()))))

print("\n=== 实验4：3D Batch Shape ===")
X3 = torch.arange(2*3*4, dtype=torch.float32).reshape(2, 3, 4) / 10
Q3 = q_proj(X3)
K3 = k_proj(X3)
V3 = v_proj(X3)
print("X3.shape =", tuple(X3.shape))
print("Q3.shape =", tuple(Q3.shape))
print("K3.shape =", tuple(K3.shape))
print("V3.shape =", tuple(V3.shape))

print("\n=== 实验5：Projection 阶段不会混合 Token ===")
X_changed = X.clone()
X_changed[0, 0] += 10.0
Q_before = q_proj(X).detach()
Q_after = q_proj(X_changed).detach()
row_diff = torch.max(torch.abs(Q_after - Q_before), dim=1).values
print("只修改 Token1 后，各 Token 的 Q 最大变化 =", row_diff.tolist())
print("结论：只有被修改的 Token 那一行 Q 变化，其余 Token 不变。")

print("\n=== 实验6：三套独立 Projection 与 fused QKV 等价 ===")
# 构造一个一次输出 9 维的 Linear，相当于把 Q/K/V 三套权重按输出维拼起来。
fused = nn.Linear(4, 9, bias=False)
with torch.no_grad():
    fused.weight.copy_(torch.cat([
        torch.tensor(WQ_np.T),
        torch.tensor(WK_np.T),
        torch.tensor(WV_np.T)
    ], dim=0))

QKV = fused(X)
Q_f, K_f, V_f = torch.chunk(QKV, 3, dim=-1)
print("fused output shape =", tuple(QKV.shape))
print("Q fused 最大误差 =", float(torch.max(torch.abs(Q_f - Q_t))))
print("K fused 最大误差 =", float(torch.max(torch.abs(K_f - K_t))))
print("V fused 最大误差 =", float(torch.max(torch.abs(V_f - V_t))))

print("\n=== 实验7：Q @ K^T 只做 Shape 预览 ===")
scores = Q_t @ K_t.transpose(-2, -1)
print("Q.shape =", tuple(Q_t.shape))
print("K^T.shape =", tuple(K_t.transpose(-2, -1).shape))
print("QK^T.shape =", tuple(scores.shape))
print("QK^T =\n", scores.detach().numpy())

print("\n=== 实验8：三条分支都有自己的 Gradient，同时共享 X ===")
X_grad = torch.tensor(X_np, requires_grad=True)
qg = nn.Linear(4, 3, bias=False)
kg = nn.Linear(4, 3, bias=False)
vg = nn.Linear(4, 3, bias=False)
with torch.no_grad():
    qg.weight.copy_(torch.tensor(WQ_np.T))
    kg.weight.copy_(torch.tensor(WK_np.T))
    vg.weight.copy_(torch.tensor(WV_np.T))

Qg = qg(X_grad)
Kg = kg(X_grad)
Vg = vg(X_grad)

# 人为定义一个简单标量 Loss，使三条分支都参与 Backward。
loss = (Qg**2).mean() + 0.5*(Kg**2).mean() + 0.25*(Vg**2).mean()
loss.backward()

print("loss =", float(loss))
print("||grad W_Q|| =", float(qg.weight.grad.norm()))
print("||grad W_K|| =", float(kg.weight.grad.norm()))
print("||grad W_V|| =", float(vg.weight.grad.norm()))
print("||grad X||   =", float(X_grad.grad.norm()))
print("X 的梯度同时接收 Q/K/V 三条分支回传的贡献。")

print("\n=== 实验9：参数量 ===")
D = 768
weights_only = 3 * D * D
with_bias = weights_only + 3 * D
print("D_model=768 时，三套 D->D QKV weight 参数量 =", weights_only)
print("若三套都有 bias，总参数量 =", with_bias)
print("一个 Linear(768, 2304) 的参数量与三套 Linear(768,768) 完全相同。")
