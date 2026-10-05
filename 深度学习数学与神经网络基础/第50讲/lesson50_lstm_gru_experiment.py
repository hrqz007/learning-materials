# -*- coding: utf-8 -*-
"""
第50讲实验：LSTM 与 GRU
目标：
1. 手算一个 LSTMCell，并与 PyTorch 对照；
2. 观察 Forget Gate 对长期记忆与梯度的影响；
3. 比较普通 RNN 与 LSTM 直接 Cell-State 路径的梯度保留；
4. 手算一个 GRUCell，并与 PyTorch 对照；
5. 比较 RNN / GRU / LSTM 参数量；
6. 验证 LSTM / GRU 的输入输出 Shape；
7. 验证分块流式处理时，只要正确传递状态，结果可与整段处理一致。
"""

from pathlib import Path
import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 固定随机种子，便于复现实验。
torch.manual_seed(50)
np.random.seed(50)

# 输出目录。
BASE = Path(__file__).resolve().parent
FIG_DIR = BASE / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = BASE / "第50讲_实验实际输出.txt"

# 使用容器中可用的中文字体，避免图像标题乱码。
plt.rcParams["font.family"] = ["Noto Sans CJK JP"]
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP"]
plt.rcParams["axes.unicode_minus"] = False


def sigmoid(x):
    """标量 Sigmoid。"""
    return 1.0 / (1.0 + math.exp(-x))


lines = []


def log(text=""):
    """同时打印并记录实验输出。"""
    print(text)
    lines.append(str(text))


# ============================================================
# 实验1：手算一个 LSTMCell，并与 PyTorch 对照
# ============================================================
log("=" * 72)
log("实验1：手算一个 LSTMCell，并与 PyTorch nn.LSTMCell 对照")
log("=" * 72)

# 当前输入、上一 Hidden State、上一 Cell State。
x = 0.7
h_prev = 0.2
c_prev = 0.5

# 四组 Gate / Candidate 的参数。
# PyTorch LSTMCell 的 Gate 顺序是 i, f, g, o。
wi_x, wi_h, bi = 0.8, 0.3, -0.1
wf_x, wf_h, bf = -0.4, 0.5, 1.0
wg_x, wg_h, bg = 0.6, -0.2, 0.0
wo_x, wo_h, bo = 0.7, 0.1, -0.2

# Input Gate：决定新候选信息写入多少。
i = sigmoid(wi_x * x + wi_h * h_prev + bi)
# Forget Gate：决定旧 Cell State 保留多少。
f = sigmoid(wf_x * x + wf_h * h_prev + bf)
# Candidate：候选写入内容，范围约在 [-1, 1]。
g = math.tanh(wg_x * x + wg_h * h_prev + bg)
# Output Gate：决定 Cell State 的哪些信息暴露为 Hidden State。
o = sigmoid(wo_x * x + wo_h * h_prev + bo)
# Cell State：旧记忆保留项 + 新记忆写入项。
c = f * c_prev + i * g
# Hidden State：Cell State 经过 tanh 后，再由 Output Gate 控制暴露比例。
h = o * math.tanh(c)

log(f"输入 x={x:.4f}, h_prev={h_prev:.4f}, c_prev={c_prev:.4f}")
log(f"Input gate i   = {i:.6f}")
log(f"Forget gate f  = {f:.6f}")
log(f"Candidate g    = {g:.6f}")
log(f"Output gate o  = {o:.6f}")
log(f"New cell c     = {c:.6f}")
log(f"New hidden h   = {h:.6f}")

# 构造 PyTorch LSTMCell，并把参数人工设置成与上面的手算完全一致。
lstm_cell = nn.LSTMCell(input_size=1, hidden_size=1)
with torch.no_grad():
    # weight_ih / weight_hh 的四行顺序为 i, f, g, o。
    lstm_cell.weight_ih.copy_(torch.tensor([[wi_x], [wf_x], [wg_x], [wo_x]], dtype=torch.float32))
    lstm_cell.weight_hh.copy_(torch.tensor([[wi_h], [wf_h], [wg_h], [wo_h]], dtype=torch.float32))
    # 为方便手算，把总 Bias 全放到 bias_ih，bias_hh 设为 0。
    lstm_cell.bias_ih.copy_(torch.tensor([bi, bf, bg, bo], dtype=torch.float32))
    lstm_cell.bias_hh.zero_()

x_t = torch.tensor([[x]], dtype=torch.float32)
h0_t = torch.tensor([[h_prev]], dtype=torch.float32)
c0_t = torch.tensor([[c_prev]], dtype=torch.float32)
h_t, c_t = lstm_cell(x_t, (h0_t, c0_t))

log(f"PyTorch c      = {c_t.item():.6f}")
log(f"PyTorch h      = {h_t.item():.6f}")
log(f"|manual c - torch c| = {abs(c - c_t.item()):.3e}")
log(f"|manual h - torch h| = {abs(h - h_t.item()):.3e}")
log()

# 绘制 Gate 与状态的数值图。
labels = ["Input i", "Forget f", "Candidate g", "Output o", "Cell c", "Hidden h"]
values = [i, f, g, o, c, h]
plt.figure(figsize=(8, 4.5))
plt.bar(labels, values)
plt.axhline(0, linewidth=1)
plt.ylabel("数值")
plt.title("LSTM 单步：Gate 与状态的实际数值")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(FIG_DIR / "lesson50_lstm_gate_values.png", dpi=180)
plt.close()


# ============================================================
# 实验2：Forget Gate 决定直接 Cell-State 路径保留多少
# ============================================================
log("=" * 72)
log("实验2：Forget Gate 与长期记忆保留")
log("=" * 72)

steps = np.arange(0, 61)
forget_values = [0.5, 0.9, 0.99, 0.999]
plt.figure(figsize=(8, 5))
for fv in forget_values:
    retained = fv ** steps
    plt.plot(steps, retained, label=f"f={fv}")
    log(f"f={fv:>5}: 10步={fv**10:.6f}, 30步={fv**30:.6f}, 60步={fv**60:.6f}")
plt.xlabel("时间步 T")
plt.ylabel(r"直接 Cell-State 路径保留比例 $f^T$")
plt.title("Forget Gate 越接近 1，旧 Cell State 衰减越慢")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(FIG_DIR / "lesson50_forget_gate_retention.png", dpi=180)
plt.close()
log()


# ============================================================
# 实验3：普通 RNN 与 LSTM 直接 Cell-State 路径的梯度保留
# ============================================================
log("=" * 72)
log("实验3：普通 RNN 与 LSTM 直接记忆路径的梯度保留")
log("=" * 72)

T_values = [1, 5, 10, 20, 40, 80]
rnn_grads = []
lstm_grads = []
for T in T_values:
    # 普通 RNN：h_t = tanh(1.2 * h_{t-1})。
    h0 = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    h_cur = h0
    for _ in range(T):
        h_cur = torch.tanh(1.2 * h_cur)
    h_cur.backward()
    grad_rnn = h0.grad.item()

    # 简化的 LSTM 直接 Cell-State 路径：c_t = 0.99 * c_{t-1}。
    # 这里故意只隔离“直接 Cell State 路径”，用来说明为什么它更容易保留梯度。
    c0 = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    c_cur = c0
    for _ in range(T):
        c_cur = 0.99 * c_cur
    c_cur.backward()
    grad_lstm = c0.grad.item()

    rnn_grads.append(abs(grad_rnn))
    lstm_grads.append(abs(grad_lstm))
    log(f"T={T:>2}: |d h_T/d h_0|={abs(grad_rnn):.6e}, |d c_T/d c_0|={abs(grad_lstm):.6e}")

plt.figure(figsize=(8, 5))
plt.semilogy(T_values, rnn_grads, marker="o", label="Vanilla RNN: tanh(1.2 h)")
plt.semilogy(T_values, lstm_grads, marker="o", label="LSTM direct cell path: f=0.99")
plt.xlabel("时间步 T")
plt.ylabel("初始状态到最终状态的梯度绝对值（对数轴）")
plt.title("LSTM 的直接 Cell-State 路径可以显著延缓梯度衰减")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(FIG_DIR / "lesson50_rnn_vs_lstm_gradient.png", dpi=180)
plt.close()
log("注意：真实 LSTM 的总梯度还包含 Gate 依赖 h_{t-1} 的其他路径；这里隔离的是最重要的直接 Cell-State 路径。")
log()


# ============================================================
# 实验4：手算一个 GRUCell，并与 PyTorch 对照
# ============================================================
log("=" * 72)
log("实验4：手算一个 GRUCell，并与 PyTorch nn.GRUCell 对照")
log("=" * 72)

x = 0.7
h_prev = 0.2

# PyTorch GRUCell Gate 顺序是 r, z, n。
wr_x, wr_h, br_i, br_h = 0.5, 0.4, -0.1, 0.0
wz_x, wz_h, bz_i, bz_h = -0.3, 0.6, 0.2, 0.0
wn_x, wn_h, bn_i, bn_h = 0.8, -0.5, 0.1, -0.05

# Reset Gate。
r = sigmoid(wr_x * x + br_i + wr_h * h_prev + br_h)
# Update Gate。
z = sigmoid(wz_x * x + bz_i + wz_h * h_prev + bz_h)
# Candidate Hidden。这里严格按照 PyTorch GRUCell 的定义：
# n = tanh(W_in x + b_in + r * (W_hn h + b_hn))。
n = math.tanh(wn_x * x + bn_i + r * (wn_h * h_prev + bn_h))
# 新 Hidden：z 越大，越保留旧 Hidden；z 越小，越采用新 Candidate。
h_new = (1.0 - z) * n + z * h_prev

log(f"Reset gate r   = {r:.6f}")
log(f"Update gate z  = {z:.6f}")
log(f"Candidate n    = {n:.6f}")
log(f"New hidden h   = {h_new:.6f}")

gru_cell = nn.GRUCell(input_size=1, hidden_size=1)
with torch.no_grad():
    gru_cell.weight_ih.copy_(torch.tensor([[wr_x], [wz_x], [wn_x]], dtype=torch.float32))
    gru_cell.weight_hh.copy_(torch.tensor([[wr_h], [wz_h], [wn_h]], dtype=torch.float32))
    gru_cell.bias_ih.copy_(torch.tensor([br_i, bz_i, bn_i], dtype=torch.float32))
    gru_cell.bias_hh.copy_(torch.tensor([br_h, bz_h, bn_h], dtype=torch.float32))

x_t = torch.tensor([[x]], dtype=torch.float32)
h0_t = torch.tensor([[h_prev]], dtype=torch.float32)
h_t = gru_cell(x_t, h0_t)

log(f"PyTorch h      = {h_t.item():.6f}")
log(f"|manual h - torch h| = {abs(h_new - h_t.item()):.3e}")
log()

plt.figure(figsize=(7, 4.5))
plt.bar(["Reset r", "Update z", "Candidate n", "Hidden h"], [r, z, n, h_new])
plt.axhline(0, linewidth=1)
plt.ylabel("数值")
plt.title("GRU 单步：Gate 与 Hidden State")
plt.tight_layout()
plt.savefig(FIG_DIR / "lesson50_gru_gate_values.png", dpi=180)
plt.close()


# ============================================================
# 实验5：RNN / GRU / LSTM 参数量比较
# ============================================================
log("=" * 72)
log("实验5：RNN / GRU / LSTM 参数量比较")
log("=" * 72)

D = 128
H = 256
rnn = nn.RNN(D, H, batch_first=True)
gru = nn.GRU(D, H, batch_first=True)
lstm = nn.LSTM(D, H, batch_first=True)

count = lambda m: sum(p.numel() for p in m.parameters())
rnn_params = count(rnn)
gru_params = count(gru)
lstm_params = count(lstm)

log(f"input_size={D}, hidden_size={H}")
log(f"RNN  参数量 = {rnn_params:,}")
log(f"GRU  参数量 = {gru_params:,}")
log(f"LSTM 参数量 = {lstm_params:,}")
log(f"GRU / RNN   = {gru_params / rnn_params:.2f}x")
log(f"LSTM / RNN  = {lstm_params / rnn_params:.2f}x")
log()

plt.figure(figsize=(7, 4.5))
plt.bar(["RNN", "GRU", "LSTM"], [rnn_params, gru_params, lstm_params])
plt.ylabel("参数量")
plt.title(f"相同 input_size={D}, hidden_size={H} 时的参数量")
plt.tight_layout()
plt.savefig(FIG_DIR / "lesson50_parameter_count.png", dpi=180)
plt.close()


# ============================================================
# 实验6：PyTorch LSTM / GRU 的 Shape Flow
# ============================================================
log("=" * 72)
log("实验6：PyTorch LSTM / GRU 的输入输出 Shape")
log("=" * 72)

B, T, D, H = 2, 5, 4, 6
X = torch.randn(B, T, D)

lstm = nn.LSTM(input_size=D, hidden_size=H, batch_first=True)
gru = nn.GRU(input_size=D, hidden_size=H, batch_first=True)

lstm_out, (h_n_lstm, c_n_lstm) = lstm(X)
gru_out, h_n_gru = gru(X)

log(f"Input X shape      = {tuple(X.shape)}")
log(f"LSTM output shape  = {tuple(lstm_out.shape)}")
log(f"LSTM h_n shape     = {tuple(h_n_lstm.shape)}")
log(f"LSTM c_n shape     = {tuple(c_n_lstm.shape)}")
log(f"GRU output shape   = {tuple(gru_out.shape)}")
log(f"GRU h_n shape      = {tuple(h_n_gru.shape)}")
log()

# 用 Matplotlib 画一个简单的 Shape Flow 图。
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis("off")
boxes = [
    (0.04, 0.58, "Input\n(B,T,D)\n(2,5,4)"),
    (0.27, 0.58, "LSTM / GRU\n时间递推"),
    (0.53, 0.70, "Output\n(B,T,H)\n(2,5,6)"),
    (0.78, 0.78, "h_n\n(1,B,H)\n(1,2,6)"),
    (0.78, 0.43, "LSTM 额外 c_n\n(1,B,H)\n(1,2,6)"),
]
for x0, y0, txt in boxes:
    ax.text(x0, y0, txt, ha="center", va="center", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="black"))
arrow = dict(arrowstyle="->", linewidth=1.6)
ax.annotate("", xy=(0.22, 0.58), xytext=(0.13, 0.58), arrowprops=arrow)
ax.annotate("", xy=(0.47, 0.67), xytext=(0.36, 0.60), arrowprops=arrow)
ax.annotate("", xy=(0.72, 0.77), xytext=(0.62, 0.72), arrowprops=arrow)
ax.annotate("", xy=(0.72, 0.47), xytext=(0.62, 0.64), arrowprops=arrow)
ax.set_title("LSTM / GRU 的基本 Shape Flow", fontsize=14)
plt.tight_layout()
plt.savefig(FIG_DIR / "lesson50_shape_flow.png", dpi=180)
plt.close()


# ============================================================
# 实验7：分块流式处理与状态传递
# ============================================================
log("=" * 72)
log("实验7：分块流式处理，只要正确传递状态，结果与整段处理一致")
log("=" * 72)

# 为了排除 Dropout 影响，使用单层默认 dropout=0 的 LSTM / GRU。
torch.manual_seed(123)
X = torch.randn(2, 5, 4)
lstm_stream = nn.LSTM(4, 6, batch_first=True)
gru_stream = nn.GRU(4, 6, batch_first=True)

# LSTM：整段一次处理。
full_lstm_out, (full_lstm_h, full_lstm_c) = lstm_stream(X)
# LSTM：先处理前 2 个时间步，再把状态传给后 3 个时间步。
out1, state1 = lstm_stream(X[:, :2, :])
out2, state2 = lstm_stream(X[:, 2:, :], state1)
chunk_lstm_out = torch.cat([out1, out2], dim=1)

# GRU：整段一次处理。
full_gru_out, full_gru_h = gru_stream(X)
# GRU：分块处理并传递 Hidden State。
gout1, gh1 = gru_stream(X[:, :2, :])
gout2, gh2 = gru_stream(X[:, 2:, :], gh1)
chunk_gru_out = torch.cat([gout1, gout2], dim=1)

lstm_out_diff = (full_lstm_out - chunk_lstm_out).abs().max().item()
lstm_h_diff = (full_lstm_h - state2[0]).abs().max().item()
lstm_c_diff = (full_lstm_c - state2[1]).abs().max().item()
gru_out_diff = (full_gru_out - chunk_gru_out).abs().max().item()
gru_h_diff = (full_gru_h - gh2).abs().max().item()

log(f"LSTM full vs chunk output max diff = {lstm_out_diff:.3e}")
log(f"LSTM final h max diff              = {lstm_h_diff:.3e}")
log(f"LSTM final c max diff              = {lstm_c_diff:.3e}")
log(f"GRU full vs chunk output max diff  = {gru_out_diff:.3e}")
log(f"GRU final h max diff               = {gru_h_diff:.3e}")
log()

plt.figure(figsize=(7.5, 4.5))
labels = ["LSTM output", "LSTM h", "LSTM c", "GRU output", "GRU h"]
vals = [lstm_out_diff, lstm_h_diff, lstm_c_diff, gru_out_diff, gru_h_diff]
plt.bar(labels, vals)
plt.yscale("symlog", linthresh=1e-12)
plt.ylabel("最大绝对差异")
plt.title("整段处理 vs 分块处理 + 状态传递")
plt.xticks(rotation=18)
plt.tight_layout()
plt.savefig(FIG_DIR / "lesson50_streaming_state.png", dpi=180)
plt.close()


# ============================================================
# 总结
# ============================================================
log("=" * 72)
log("实验总结")
log("=" * 72)
log("1. LSTM 的核心不是简单‘多几个门’，而是增加了独立 Cell State 和记忆写入/保留/读出控制。")
log("2. Forget Gate 接近 1 时，直接 Cell-State 路径能显著延缓信息与梯度衰减，但并不保证永不消失。")
log("3. GRU 用更紧凑的 Update/Reset Gate 合并了部分 LSTM 机制，没有独立 Cell State。")
log("4. RNN/GRU/LSTM 都沿时间递推，因此训练仍存在时间轴串行依赖。")
log("5. LSTM/GRU 的状态可以跨 chunk 传递，所以天然支持流式/在线序列处理。")

# 写入实际输出文件。
OUT_FILE.write_text("\n".join(lines), encoding="utf-8")
