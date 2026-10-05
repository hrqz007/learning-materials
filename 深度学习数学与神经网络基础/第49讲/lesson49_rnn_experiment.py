# -*- coding: utf-8 -*-
"""第49讲：Sequence 与 RNN 实验
目标：用最小数值实验理解 Hidden State、时间共享参数、PyTorch nn.RNN 与 BPTT 梯度路径。
"""

from pathlib import Path
import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).resolve().parent
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

np.set_printoptions(precision=6, suppress=True)
torch.set_printoptions(precision=6, sci_mode=False)

torch.manual_seed(7)
np.random.seed(7)


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def scalar_rnn_numpy(xs, wx=0.8, wh=0.5, b=-0.1, h0=0.0):
    """最小标量 RNN：h_t=tanh(wx*x_t + wh*h_{t-1}+b)。"""
    hs = []
    h = float(h0)
    for t, x in enumerate(xs, start=1):
        z = wx * float(x) + wh * h + b
        h = math.tanh(z)
        hs.append(h)
        print(f"t={t}: x={float(x): .3f}, z={z: .6f}, h={h: .6f}")
    return np.array(hs, dtype=np.float64)


section("实验1：手工递推——Hidden State 是怎样一步一步更新的")
xs = np.array([1.0, 2.0, -1.0], dtype=np.float64)
hs = scalar_rnn_numpy(xs)
print("输入序列 xs =", xs)
print("全部 hidden states =", hs)
print("最终 hidden state =", hs[-1])


section("实验2：同一个当前输入，因为历史不同，Hidden State 仍可不同")
# 两条序列的最后一个输入都等于 0.5，但前面的历史不同
seq_a = np.array([2.0, 2.0, 0.5], dtype=np.float64)
seq_b = np.array([-2.0, -2.0, 0.5], dtype=np.float64)
h_a = scalar_rnn_numpy(seq_a)
h_b = scalar_rnn_numpy(seq_b)
print("A 最后输入 =", seq_a[-1], "最终 h =", h_a[-1])
print("B 最后输入 =", seq_b[-1], "最终 h =", h_b[-1])
print("同一当前输入下最终 hidden 差值 =", abs(h_a[-1] - h_b[-1]))


section("实验3：手工递推与 PyTorch nn.RNN 完全对齐")
# 使用 batch_first=True。固定权重，便于与手工公式逐步核对。
rnn = nn.RNN(input_size=1, hidden_size=1, num_layers=1, nonlinearity="tanh", batch_first=True, bias=True)
with torch.no_grad():
    rnn.weight_ih_l0.fill_(0.8)
    rnn.weight_hh_l0.fill_(0.5)
    # PyTorch RNN 有两组 bias，公式里它们相加。设一个为 -0.1，另一个为 0。
    rnn.bias_ih_l0.fill_(-0.1)
    rnn.bias_hh_l0.zero_()

x_torch = torch.tensor(xs, dtype=torch.float32).view(1, 3, 1)
h0 = torch.zeros(1, 1, 1)
out, h_n = rnn(x_torch, h0)
manual = hs.astype(np.float32).reshape(1, 3, 1)
print("PyTorch input shape =", tuple(x_torch.shape))
print("PyTorch output shape =", tuple(out.shape))
print("PyTorch h_n shape =", tuple(h_n.shape))
print("PyTorch output =", out.detach().numpy().reshape(-1))
print("Manual output  =", manual.reshape(-1))
print("max abs diff   =", float(np.max(np.abs(out.detach().numpy() - manual))))


section("实验4：RNN 参数量与 Sequence Length 无关——参数在时间维共享")
input_size = 4
hidden_size = 6
rnn2 = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)
param_count = sum(p.numel() for p in rnn2.parameters())
for T in [5, 20, 100]:
    x = torch.randn(2, T, input_size)
    y, h = rnn2(x)
    print(f"T={T:3d}: input={tuple(x.shape)}, output={tuple(y.shape)}, h_n={tuple(h.shape)}, params={param_count}")
print("理论参数量 = H*D + H*H + 2*H =", hidden_size*input_size + hidden_size*hidden_size + 2*hidden_size)


section("实验5：BPTT 最小模型——时间路径上的连续乘积")
T_values = np.arange(1, 31)
a_values = [0.5, 0.9, 1.0, 1.1, 1.2]
for a in a_values:
    grad = a ** T_values[-1]
    print(f"a={a:3.1f}, |dh_T/dh_0| at T=30 = {grad:.10g}")

plt.figure(figsize=(8, 5))
for a in a_values:
    plt.semilogy(T_values, np.abs(a ** T_values), label=f"a={a}")
plt.xlabel("Time steps T")
plt.ylabel("|dh_T / dh_0| = |a|^T (log scale)")
plt.title("Temporal gradient product in a minimal recurrence")
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "lesson49_gradient_through_time.png", dpi=180)
plt.close()


section("实验6：tanh 的局部导数会进一步压缩长期梯度")
# 固定递推 h_t=tanh(a*h_{t-1})，用 autograd 直接看 dh_T/dh_0。
def tanh_temporal_grad(a, h0_value, steps):
    h0 = torch.tensor(float(h0_value), requires_grad=True)
    h = h0
    for _ in range(steps):
        h = torch.tanh(torch.tensor(float(a)) * h)
    h.backward()
    return float(h.detach()), float(h0.grad)

for h0_value in [0.1, 1.0, 3.0]:
    hT, grad = tanh_temporal_grad(1.0, h0_value, 20)
    print(f"h0={h0_value:.1f}: h20={hT:.8f}, dh20/dh0={grad:.10g}")


section("实验7：RNN Shape——(B,T,D) -> (B,T,H)，h_n 只保留各层最终 Hidden")
B, T, D, H = 3, 7, 5, 4
rnn3 = nn.RNN(D, H, num_layers=2, batch_first=True)
x = torch.randn(B, T, D)
y, h_n = rnn3(x)
print("x.shape   =", tuple(x.shape))
print("y.shape   =", tuple(y.shape))
print("h_n.shape =", tuple(h_n.shape))
print("解释：y 保存每个时间步的 top-layer hidden；h_n 保存每一层最后时间步 hidden。")


section("实验8：Many-to-one 的最小前向——用最后 Hidden 做分类")
classifier = nn.Sequential(
    nn.Linear(H, 2)
)
logits = classifier(h_n[-1])
print("最后一层 h_n[-1].shape =", tuple(h_n[-1].shape))
print("分类 logits.shape       =", tuple(logits.shape))


# 图1：手工 hidden state 演化
plt.figure(figsize=(7, 4))
plt.plot(np.arange(1, len(hs)+1), hs, marker="o")
plt.axhline(0, linewidth=0.8)
plt.xlabel("Time step")
plt.ylabel("Hidden state h_t")
plt.title("Hidden state evolution for x=[1,2,-1]")
plt.tight_layout()
plt.savefig(FIG_DIR / "lesson49_hidden_state_evolution.png", dpi=180)
plt.close()

# 图2：两段不同历史，同一末尾输入
plt.figure(figsize=(7, 4))
steps = np.arange(1, 4)
plt.plot(steps, h_a, marker="o", label="history A: [2,2,0.5]")
plt.plot(steps, h_b, marker="o", label="history B: [-2,-2,0.5]")
plt.xlabel("Time step")
plt.ylabel("Hidden state")
plt.title("Same current input, different history, different hidden state")
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "lesson49_history_changes_hidden.png", dpi=180)
plt.close()

# 图3：参数量 vs sequence length
Ts = np.array([1, 5, 10, 20, 50, 100])
params = np.full_like(Ts, param_count)
plt.figure(figsize=(7, 4))
plt.plot(Ts, params, marker="o")
plt.xlabel("Sequence length T")
plt.ylabel("Parameter count")
plt.title("RNN parameter count does not grow with sequence length")
plt.tight_layout()
plt.savefig(FIG_DIR / "lesson49_parameter_sharing.png", dpi=180)
plt.close()

# 图4：RNN shape flow
fig, ax = plt.subplots(figsize=(9, 3.2))
ax.axis("off")
labels = ["Input\n(B,T,D)", "Shared RNN Cell\nrepeated T times", "Outputs\n(B,T,H)", "Final hidden\n(L,B,H)"]
xs_pos = [0.08, 0.36, 0.67, 0.90]
for x0, lab in zip(xs_pos, labels):
    ax.text(x0, 0.5, lab, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black"), transform=ax.transAxes)
for a, b in zip(xs_pos[:-1], xs_pos[1:]):
    ax.annotate("", xy=(b-0.07, 0.5), xytext=(a+0.08, 0.5), xycoords=ax.transAxes, arrowprops=dict(arrowstyle="->"))
ax.set_title("RNN tensor shape flow")
plt.tight_layout()
plt.savefig(FIG_DIR / "lesson49_rnn_shape_flow.png", dpi=180)
plt.close()

print("\n已生成图像：")
for p in sorted(FIG_DIR.glob("lesson49_*.png")):
    print(" -", p.name)
