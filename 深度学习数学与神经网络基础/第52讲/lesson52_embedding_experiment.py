# -*- coding: utf-8 -*-
"""
第52讲实验：Embedding 的深度学习本质

目标：
1. 验证 Embedding lookup 与 one-hot @ EmbeddingMatrix 数学等价。
2. 验证 (B, T) Token IDs -> (B, T, D) Embeddings 的 Shape 变化。
3. 观察 nn.Embedding 的梯度：只更新被访问的词表行，重复 Token 的梯度会累加。
4. 验证 padding_idx 可以阻止 Padding 行参与梯度更新。
5. 说明 Token Embedding 本身不携带位置：同一个 Token 在不同位置查到同一个向量。
6. 演示一个最小训练任务：Embedding row 本身就是可训练 Parameter。

说明：
- 代码尽量使用小数字，便于和讲义手算对照。
- 默认 CPU 即可运行。
"""

from pathlib import Path
import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---------------------------
# 0. 固定随机种子与输出目录
# ---------------------------
torch.manual_seed(52)
np.random.seed(52)

OUT_DIR = Path(__file__).resolve().parent
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 尝试使用系统中的中文字体；如果不可用，matplotlib 会回退。
CJK_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(CJK_FONT)
CJK_NAME = font_manager.FontProperties(fname=CJK_FONT).get_name()
plt.rcParams["font.family"] = CJK_NAME
plt.rcParams["axes.unicode_minus"] = False


def print_title(title: str):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ------------------------------------------------------------
# 实验1：Embedding lookup 与 one-hot @ E 是否完全等价？
# ------------------------------------------------------------
print_title("实验1：Embedding lookup 与 one-hot @ EmbeddingMatrix 的等价性")

# 词表大小 V=5，每个 Token 的向量维度 D=3。
E = torch.tensor(
    [
        [0.10, 0.20, 0.30],   # Token 0
        [1.00, 0.00, -1.00],  # Token 1
        [0.50, 0.50, 0.50],   # Token 2
        [-0.20, 0.40, 0.80],  # Token 3
        [2.00, 1.00, 0.00],   # Token 4
    ],
    dtype=torch.float32,
)

# 假设当前 Token ID 是 3。
token_id = 3

# 方法A：直接查 Embedding Matrix 的第3行。
lookup = E[token_id]

# 方法B：先把 Token ID 3 写成 one-hot [0,0,0,1,0]，再乘 E。
one_hot = torch.nn.functional.one_hot(torch.tensor(token_id), num_classes=E.shape[0]).float()
via_one_hot = one_hot @ E

print("Embedding Matrix E.shape:", tuple(E.shape))
print("token_id:", token_id)
print("直接 lookup:", lookup.tolist())
print("one-hot:", one_hot.tolist())
print("one-hot @ E:", via_one_hot.tolist())
print("最大绝对误差:", float((lookup - via_one_hot).abs().max()))

# 图1：Embedding Matrix，并高亮被查询的行。
fig, ax = plt.subplots(figsize=(7, 4.5))
im = ax.imshow(E.numpy(), aspect="auto")
ax.set_title("Embedding Matrix：Token ID 3 查第 3 行")
ax.set_xlabel("Embedding Dimension")
ax.set_ylabel("Token ID")
ax.set_xticks(range(E.shape[1]))
ax.set_yticks(range(E.shape[0]))
for i in range(E.shape[0]):
    for j in range(E.shape[1]):
        ax.text(j, i, f"{E[i, j].item():.1f}", ha="center", va="center")
# 用矩形边框标记第 token_id 行，不指定颜色，让 matplotlib 用默认配色。
from matplotlib.patches import Rectangle
ax.add_patch(Rectangle((-0.5, token_id - 0.5), E.shape[1], 1, fill=False, linewidth=2.5))
fig.colorbar(im, ax=ax, shrink=0.8)
fig.tight_layout()
fig.savefig(FIG_DIR / "lesson52_embedding_lookup.png", dpi=180)
plt.close(fig)


# ------------------------------------------------------------
# 实验2：(B,T) -> (B,T,D) 的 Shape Flow
# ------------------------------------------------------------
print_title("实验2：(B,T) Token IDs -> (B,T,D) Embeddings")

# 创建一个真正的 nn.Embedding，并把 weight 改成上面的已知矩阵。
embedding = nn.Embedding(num_embeddings=5, embedding_dim=3)
with torch.no_grad():
    embedding.weight.copy_(E)

# B=2 条序列，T=4 个 Token。
token_ids = torch.tensor(
    [
        [1, 3, 2, 1],
        [4, 0, 3, 2],
    ],
    dtype=torch.long,
)

emb = embedding(token_ids)
print("token_ids.shape:", tuple(token_ids.shape))
print("embeddings.shape:", tuple(emb.shape))
print("第0条序列第1个位置的 Token ID:", int(token_ids[0, 1]))
print("对应向量:", emb[0, 1].tolist())
print("是否等于 E[3]:", bool(torch.allclose(emb[0, 1], E[3])))

# 图2：Shape Flow。
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.axis("off")
ax.text(0.08, 0.55, "Token IDs\n(B,T) = (2,4)", ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.5", fill=False, linewidth=1.8), fontsize=12)
ax.annotate("Embedding lookup\n每个 ID 查一行", xy=(0.64, 0.55), xytext=(0.36, 0.55),
            arrowprops=dict(arrowstyle="->", linewidth=2), ha="center", va="center", fontsize=11)
ax.text(0.84, 0.55, "Embeddings\n(B,T,D) = (2,4,3)", ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.5", fill=False, linewidth=1.8), fontsize=12)
ax.set_title("Embedding 的 Shape 变化", fontsize=14)
fig.tight_layout()
fig.savefig(FIG_DIR / "lesson52_shape_flow.png", dpi=180)
plt.close(fig)


# ------------------------------------------------------------
# 实验3：只更新被访问行；重复 Token 的梯度会累加
# ------------------------------------------------------------
print_title("实验3：Embedding Gradient 的稀疏行结构与重复 Token 梯度累加")

# 使用一个全0初始化的 Embedding，便于只观察 gradient。
emb_grad = nn.Embedding(num_embeddings=6, embedding_dim=4)
with torch.no_grad():
    emb_grad.weight.zero_()

# Token 1 出现两次，Token 3 出现一次。
ids = torch.tensor([1, 3, 1], dtype=torch.long)
out = emb_grad(ids)

# 令 loss 为所有输出元素之和。
# 因为 d(sum)/d(out)=1，所以每访问一次某 Token，对那一行贡献一个全1向量。
loss = out.sum()
loss.backward()

grad = emb_grad.weight.grad.detach().clone()
print("ids:", ids.tolist())
print("weight.grad.shape:", tuple(grad.shape))
print("完整 grad matrix:\n", grad)
print("Token 1 出现2次 -> grad row 1:", grad[1].tolist())
print("Token 3 出现1次 -> grad row 3:", grad[3].tolist())
print("未出现 Token 2 -> grad row 2:", grad[2].tolist())
print("非零梯度行:", torch.where(grad.abs().sum(dim=1) > 0)[0].tolist())

# 图3：梯度矩阵热图。
fig, ax = plt.subplots(figsize=(7, 4.5))
im = ax.imshow(grad.numpy(), aspect="auto")
ax.set_title("Embedding weight.grad：只访问 Token 1 和 3")
ax.set_xlabel("Embedding Dimension")
ax.set_ylabel("Vocabulary Row / Token ID")
ax.set_xticks(range(grad.shape[1]))
ax.set_yticks(range(grad.shape[0]))
for i in range(grad.shape[0]):
    for j in range(grad.shape[1]):
        ax.text(j, i, f"{grad[i, j].item():.0f}", ha="center", va="center")
fig.colorbar(im, ax=ax, shrink=0.8)
fig.tight_layout()
fig.savefig(FIG_DIR / "lesson52_embedding_gradient_rows.png", dpi=180)
plt.close(fig)


# ------------------------------------------------------------
# 实验4：padding_idx 不应该被普通 Embedding 训练更新
# ------------------------------------------------------------
print_title("实验4：padding_idx 如何阻止 Padding Row 接收梯度")

pad_emb = nn.Embedding(num_embeddings=6, embedding_dim=3, padding_idx=0)
with torch.no_grad():
    # 为了观察输出，把非 Padding 行赋予确定值；Padding 行保持全0（PyTorch 默认也是如此）。
    pad_emb.weight[1:].copy_(torch.arange(15, dtype=torch.float32).reshape(5, 3) / 10.0)

pad_ids = torch.tensor([[0, 2, 0, 4]], dtype=torch.long)
pad_out = pad_emb(pad_ids)
pad_loss = pad_out.sum()
pad_loss.backward()

print("pad_ids:", pad_ids.tolist())
print("Padding row weight:", pad_emb.weight[0].detach().tolist())
print("Padding row grad:", pad_emb.weight.grad[0].tolist())
print("Token 2 row grad:", pad_emb.weight.grad[2].tolist())
print("Token 4 row grad:", pad_emb.weight.grad[4].tolist())


# ------------------------------------------------------------
# 实验5：Token Embedding 本身不知道位置
# ------------------------------------------------------------
print_title("实验5：同一个 Token 在不同位置得到同一个 Token Embedding")

pos_ids = torch.tensor([[3, 1, 3, 2]], dtype=torch.long)
pos_token_emb = embedding(pos_ids)

same_token_diff = (pos_token_emb[0, 0] - pos_token_emb[0, 2]).abs().max()
print("序列 Token IDs:", pos_ids.tolist())
print("位置0的 Token 3 embedding:", pos_token_emb[0, 0].tolist())
print("位置2的 Token 3 embedding:", pos_token_emb[0, 2].tolist())
print("同一 Token 不同位置，纯 Token Embedding 最大差异:", float(same_token_diff.detach()))

# 为了演示“位置信息必须从别的机制加入”，这里用一个简单的 learned positional vector 做概念示例。
position_table = torch.tensor(
    [
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.2, 0.0, 0.0],
        [0.3, 0.0, 0.0],
    ],
    dtype=torch.float32,
)
with_position = pos_token_emb + position_table.unsqueeze(0)
positioned_diff = (with_position[0, 0] - with_position[0, 2]).abs().max()
print("加入示意性 position vector 后，位置0/2最大差异:", float(positioned_diff.detach()))
print("注意：现代 LLM 也可能不用加法位置Embedding，例如 RoPE 通常作用在 Attention 的 Q/K 上。")

# 图4：同 Token 的纯 Embedding 相同，加入位置后区分开。
fig, ax = plt.subplots(figsize=(8, 4.2))
indices = np.arange(3)
width = 0.22
v0 = pos_token_emb[0, 0].detach().numpy()
v2 = pos_token_emb[0, 2].detach().numpy()
p0 = with_position[0, 0].detach().numpy()
p2 = with_position[0, 2].detach().numpy()
ax.bar(indices - 1.5 * width, v0, width, label="Token3 @ pos0: token embedding")
ax.bar(indices - 0.5 * width, v2, width, label="Token3 @ pos2: token embedding")
ax.bar(indices + 0.5 * width, p0, width, label="Token3 @ pos0: + position")
ax.bar(indices + 1.5 * width, p2, width, label="Token3 @ pos2: + position")
ax.set_xticks(indices)
ax.set_xticklabels(["dim0", "dim1", "dim2"])
ax.set_title("Token Embedding 本身不区分位置")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR / "lesson52_token_vs_position.png", dpi=180)
plt.close(fig)


# ------------------------------------------------------------
# 实验6：Embedding row 本身是可训练 Parameter
# ------------------------------------------------------------
print_title("实验6：Embedding Matrix 的某些行如何通过训练被更新")

train_emb = nn.Embedding(num_embeddings=5, embedding_dim=2)
with torch.no_grad():
    train_emb.weight.zero_()

# 训练目标：让 Token 1 -> [1,-1]，Token 3 -> [-1,1]
train_ids = torch.tensor([1, 3, 1, 3], dtype=torch.long)
targets = torch.tensor(
    [
        [1.0, -1.0],
        [-1.0, 1.0],
        [1.0, -1.0],
        [-1.0, 1.0],
    ],
    dtype=torch.float32,
)

optimizer = torch.optim.SGD(train_emb.parameters(), lr=0.4)
loss_fn = nn.MSELoss()
loss_history = []

for step in range(20):
    optimizer.zero_grad()
    pred = train_emb(train_ids)
    train_loss = loss_fn(pred, targets)
    train_loss.backward()
    optimizer.step()
    loss_history.append(float(train_loss.item()))

print("初始 Loss (step0前向后的量级可由曲线观察)")
print("最终 Loss:", loss_history[-1])
print("Token 1 learned embedding:", train_emb.weight[1].detach().tolist())
print("Token 3 learned embedding:", train_emb.weight[3].detach().tolist())
print("从未出现 Token 2 embedding:", train_emb.weight[2].detach().tolist())

# 图5：训练 Loss。
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(range(1, len(loss_history) + 1), loss_history, marker="o", markersize=3)
ax.set_xlabel("Optimizer Step")
ax.set_ylabel("MSE Loss")
ax.set_title("Embedding Row 可以通过 Gradient Descent 学习")
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(FIG_DIR / "lesson52_embedding_training_loss.png", dpi=180)
plt.close(fig)


# ------------------------------------------------------------
# 实验7：参数量与内存的最小估算
# ------------------------------------------------------------
print_title("实验7：Embedding 参数量与内存估算")

V = 50_000
D = 4_096
params = V * D
fp32_bytes = params * 4
fp16_bytes = params * 2
print("Vocab size V:", V)
print("Embedding dim D:", D)
print("参数量 V*D:", f"{params:,}")
print("仅权重 fp32 约 GiB:", fp32_bytes / (1024 ** 3))
print("仅权重 fp16/bf16 约 GiB:", fp16_bytes / (1024 ** 3))
print("注意：训练显存还会受到 gradient、optimizer state、activation 等影响。")

print_title("实验全部完成")
print("生成图目录:", FIG_DIR)
