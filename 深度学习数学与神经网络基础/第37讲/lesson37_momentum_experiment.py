# -*- coding: utf-8 -*-
"""
第37讲实验：Momentum（动量）

目标：
1. 用手工梯度序列观察 Momentum 如何“记住”过去 Gradient。
2. 在狭长 Loss Landscape 中比较普通 SGD 与 Momentum 的参数轨迹。
3. 比较不同 momentum coefficient μ 的影响，验证“μ 越大并不一定越好”。
4. 用 PyTorch 的 torch.optim.SGD(momentum=...) 验证手写 Momentum 更新规则。
5. 建立 Momentum 与后续 Adam 一阶矩估计之间的联系。
"""

from pathlib import Path
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import torch

BASE = Path(__file__).resolve().parent
FIG_DIR = BASE / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 尽量让中文图例正常显示；若本机没有该字体，会自动回退。
font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
if font_path.exists():
    font_manager.fontManager.addfont(str(font_path))
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
else:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "PingFang SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def section(title: str):
    print("\n" + "=" * 84)
    print(title)
    print("=" * 84)


# -----------------------------------------------------------------------------
# 实验1：固定方向与交替方向的 Gradient
# -----------------------------------------------------------------------------
section("实验1：Momentum 如何记住过去 Gradient——持续方向被强化，交替方向被抵消")

mu = 0.9
persistent_gradients = [1.0] * 10
alternating_gradients = [1.0 if i % 2 == 0 else -1.0 for i in range(10)]


def momentum_buffers(gradients, mu=0.9):
    """按 PyTorch 直观形式 b_t = mu*b_(t-1) + g_t 计算 momentum buffer。"""
    buffer = 0.0
    values = []
    for g in gradients:
        buffer = mu * buffer + g
        values.append(buffer)
    return np.array(values, dtype=np.float64)


persistent_buffers = momentum_buffers(persistent_gradients, mu)
alternating_buffers = momentum_buffers(alternating_gradients, mu)

print(f"momentum coefficient μ = {mu}")
print("持续同方向 Gradient:", persistent_gradients)
print("对应 momentum buffer:", np.round(persistent_buffers, 6).tolist())
print("交替正负 Gradient:", alternating_gradients)
print("对应 momentum buffer:", np.round(alternating_buffers, 6).tolist())
print("解释：持续同方向会不断积累；频繁翻转的方向会被历史项部分抵消。")

steps = np.arange(1, 11)
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(steps, persistent_buffers, marker="o", label="持续同方向 Gradient")
ax.plot(steps, alternating_buffers, marker="o", label="正负交替 Gradient")
ax.axhline(0, linewidth=1)
ax.set_xlabel("Step")
ax.set_ylabel("Momentum buffer")
ax.set_title("Momentum 对不同 Gradient 历史的响应")
ax.grid(True, alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "lesson37_gradient_history_buffer.png", dpi=180)
plt.close(fig)

# 展开式权重：当前 g 权重 1，前一步 mu，再前一步 mu^2 ...
weights = np.array([mu ** k for k in range(15)], dtype=np.float64)
print("\n过去 Gradient 的权重（当前到更早）:", np.round(weights[:10], 6).tolist())
print(f"μ=0.9 时，约 {math.log(0.5)/math.log(mu):.2f} 步前的 Gradient 权重衰减到一半。")
print(f"常用直觉尺度 1/(1-μ) = {1/(1-mu):.1f} steps（只是记忆长度的粗略量级，不是硬边界）。")

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.bar(np.arange(len(weights)), weights)
ax.set_xlabel("距离当前多少步 k")
ax.set_ylabel("历史 Gradient 权重 μ^k")
ax.set_title("Momentum 对更久以前 Gradient 的指数衰减权重")
ax.grid(True, axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(FIG_DIR / "lesson37_history_weights.png", dpi=180)
plt.close(fig)


# -----------------------------------------------------------------------------
# 实验2：狭长 Loss Landscape
# L = 0.5*(w1^2 + 50*w2^2)
# -----------------------------------------------------------------------------
section("实验2：狭长 Loss Landscape——SGD 的横向震荡与 Momentum 的阻尼/加速")

CURVATURE = 50.0
LR = 0.04
START = np.array([6.0, 1.5], dtype=np.float64)
STEPS = 80


def loss_2d(w):
    return 0.5 * (w[0] ** 2 + CURVATURE * w[1] ** 2)


def grad_2d(w):
    return np.array([w[0], CURVATURE * w[1]], dtype=np.float64)


def run_optimizer(mu_value: float, lr=LR, steps=STEPS):
    w = START.copy()
    buffer = np.zeros_like(w)
    path = [w.copy()]
    losses = [loss_2d(w)]
    for _ in range(steps):
        g = grad_2d(w)
        buffer = mu_value * buffer + g
        w = w - lr * buffer
        path.append(w.copy())
        losses.append(loss_2d(w))
    return np.array(path), np.array(losses)


path_sgd, loss_sgd = run_optimizer(mu_value=0.0)
path_mom, loss_mom = run_optimizer(mu_value=0.9)

print(f"共同起点 w={START.tolist()}, LR={LR}, 曲率系数={CURVATURE}")
print(f"普通 SGD: 初始 Loss={loss_sgd[0]:.6f}, 80步后 Loss={loss_sgd[-1]:.6f}, final w={path_sgd[-1]}")
print(f"Momentum μ=0.9: 初始 Loss={loss_mom[0]:.6f}, 80步后 Loss={loss_mom[-1]:.6f}, final w={path_mom[-1]}")
print("在这个特定 LR 下，普通 SGD 的陡峭方向更新因子恰好接近 -1，因此会持续在谷底两边翻转。")

# 等高线和轨迹
w1 = np.linspace(-6.5, 6.5, 320)
w2 = np.linspace(-2.0, 2.0, 320)
W1, W2 = np.meshgrid(w1, w2)
Z = 0.5 * (W1 ** 2 + CURVATURE * W2 ** 2)
fig, ax = plt.subplots(figsize=(9, 6))
levels = np.geomspace(0.05, 120, 22)
ax.contour(W1, W2, Z, levels=levels, linewidths=0.8)
ax.plot(path_sgd[:, 0], path_sgd[:, 1], marker="o", markersize=2.2, label="SGD (μ=0)")
ax.plot(path_mom[:, 0], path_mom[:, 1], marker="o", markersize=2.2, label="Momentum (μ=0.9)")
ax.scatter([0], [0], marker="*", s=120, label="minimum")
ax.set_xlabel("w1（平缓方向）")
ax.set_ylabel("w2（陡峭方向）")
ax.set_title("狭长 Loss Landscape：SGD 与 Momentum 的参数轨迹")
ax.grid(True, alpha=0.15)
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "lesson37_narrow_valley_paths.png", dpi=180)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(np.maximum(loss_sgd, 1e-12), label="SGD (μ=0)")
ax.plot(np.maximum(loss_mom, 1e-12), label="Momentum (μ=0.9)")
ax.set_yscale("log")
ax.set_xlabel("Step")
ax.set_ylabel("Loss (log scale)")
ax.set_title("同一 LR 下：SGD 与 Momentum 的 Loss 曲线")
ax.grid(True, alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "lesson37_sgd_vs_momentum_loss.png", dpi=180)
plt.close(fig)


# -----------------------------------------------------------------------------
# 实验3：不同 μ 并不是越大越好
# -----------------------------------------------------------------------------
section("实验3：Momentum coefficient μ 并不是越大越好")

mu_values = [0.0, 0.5, 0.9, 0.95, 0.99]
mu_results = {}
for mu_value in mu_values:
    path, losses = run_optimizer(mu_value=mu_value)
    mu_results[mu_value] = (path, losses)
    print(
        f"μ={mu_value:>4}: step10 Loss={losses[10]:.6f}, "
        f"step40 Loss={losses[40]:.6f}, final Loss={losses[-1]:.6f}"
    )

fig, ax = plt.subplots(figsize=(9, 5.5))
for mu_value in mu_values:
    losses = mu_results[mu_value][1]
    ax.plot(np.maximum(losses, 1e-12), label=f"μ={mu_value}")
ax.set_yscale("log")
ax.set_xlabel("Step")
ax.set_ylabel("Loss (log scale)")
ax.set_title("固定 LR 时，不同 Momentum coefficient 的训练轨迹")
ax.grid(True, alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "lesson37_momentum_coefficient_comparison.png", dpi=180)
plt.close(fig)

print("结论：在这个特定例子中 μ=0.5 比 μ=0.9 更快；μ=0.99 反而保留太多惯性、收敛很差。")
print("这不是说 μ=0.5 普遍最佳，而是说明 μ 与 LR、曲率、Gradient Noise 必须一起看。")


# -----------------------------------------------------------------------------
# 实验4：手写 Momentum 与 PyTorch SGD(momentum=...) 对齐
# -----------------------------------------------------------------------------
section("实验4：手写 Momentum 与 torch.optim.SGD(momentum=...) 数值对齐")

torch.manual_seed(37)
initial = torch.tensor([2.0, -1.0], dtype=torch.float64)
lr = 0.05
mu = 0.9
steps = 8

# 手写参数与 buffer
manual_param = initial.numpy().copy()
manual_buffer = np.zeros(2, dtype=np.float64)

# PyTorch Parameter
param = torch.nn.Parameter(initial.clone())
optimizer = torch.optim.SGD([param], lr=lr, momentum=mu)

print(f"初始参数: {initial.tolist()}, lr={lr}, momentum={mu}")
for step in range(1, steps + 1):
    # 定义一个透明的二次 Loss: L = 0.5*(p1^2 + 3*p2^2)
    # 手写 Gradient = [p1, 3*p2]
    manual_grad = np.array([manual_param[0], 3.0 * manual_param[1]], dtype=np.float64)
    manual_buffer = mu * manual_buffer + manual_grad
    manual_param = manual_param - lr * manual_buffer

    optimizer.zero_grad()
    loss = 0.5 * (param[0] ** 2 + 3.0 * param[1] ** 2)
    loss.backward()
    optimizer.step()

    diff = np.max(np.abs(manual_param - param.detach().numpy()))
    print(
        f"step={step:02d} manual={np.round(manual_param, 8).tolist()} "
        f"torch={np.round(param.detach().numpy(), 8).tolist()} max_abs_diff={diff:.3e}"
    )

print("最终最大差异应接近浮点误差，说明手写公式与 PyTorch 默认 dampening=0 的 Momentum 一致。")


# -----------------------------------------------------------------------------
# 实验5：Momentum 与 Adam 第一矩之间的联系（只做概念数值演示）
# -----------------------------------------------------------------------------
section("实验5：Momentum 与 Adam 的第一矩 m_t 有什么关系")

grad_seq = np.array([2.0, 2.0, 1.0, -1.0, 0.5], dtype=np.float64)
mu = 0.9
buffer = 0.0
adam_m = 0.0
print("Gradient sequence:", grad_seq.tolist())
print("PyTorch-style momentum buffer: b_t = μ b_(t-1) + g_t")
print("Adam first moment:             m_t = β1 m_(t-1) + (1-β1) g_t")
for t, g in enumerate(grad_seq, 1):
    buffer = mu * buffer + g
    adam_m = mu * adam_m + (1 - mu) * g
    print(f"t={t}: g={g:+.3f}, momentum_buffer={buffer:+.6f}, adam_m={adam_m:+.6f}")
print("两者都在利用历史 Gradient；但尺度定义不同，Adam 还会加入 bias correction 和 second moment。")

print("\n所有实验完成。图像保存在：", FIG_DIR)
