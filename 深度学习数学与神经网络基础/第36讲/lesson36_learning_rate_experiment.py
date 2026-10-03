# -*- coding: utf-8 -*-
"""
第36讲实验：Learning Rate（学习率）
目标：
1. 在一维二次函数上观察不同学习率的收敛、振荡与发散。
2. 在不同曲率方向的二维 Loss Landscape 中理解“最陡方向限制全局学习率”。
3. 在同一初始化、同一数据上比较 PyTorch 训练的不同学习率。
4. 比较 SGD 与 Adam 在相同 nominal learning rate 下的实际更新差异。
5. 记录参数更新量与参数量级的相对比例。
"""

import math
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

try:
    import torch
    import torch.nn as nn
except Exception as exc:
    raise RuntimeError("本实验需要 PyTorch。请先安装 torch。") from exc

BASE = Path(__file__).resolve().parent
FIG_DIR = BASE / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 尽量让中文图例在当前环境可显示；若字体不存在，matplotlib 会自动回退。
_linux_cjk_font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
if _linux_cjk_font.exists():
    font_manager.fontManager.addfont(str(_linux_cjk_font))
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
else:
    # Windows/macOS 上优先尝试常见中文字体；即使字体不存在，代码仍可运行。
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "PingFang SC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def section(title: str):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def run_1d_quadratic(lr: float, steps: int = 25, w0: float = 8.0):
    """L(w)=(w-3)^2，手写 Gradient Descent。"""
    w = float(w0)
    history = []
    for step in range(steps + 1):
        loss = (w - 3.0) ** 2
        grad = 2.0 * (w - 3.0)
        history.append((step, w, loss, grad))
        if step < steps:
            w = w - lr * grad
    return history


section("实验1：一维二次函数——学习率决定一步走多远")
learning_rates = [0.01, 0.1, 0.5, 0.9, 1.0, 1.1]
summary_1d = {}
for lr in learning_rates:
    hist = run_1d_quadratic(lr, steps=25)
    summary_1d[lr] = hist
    last = hist[-1]
    print(
        f"lr={lr:>4}: final w={last[1]: .8f}, final loss={last[2]: .8e}, "
        f"first-step w={hist[1][1]: .4f}"
    )

# 理论稳定区间：L=(w-3)^2, grad=2(w-3), 误差 e_{t+1}=(1-2lr)e_t
print("\n理论：误差满足 e_(t+1) = (1 - 2*lr) * e_t")
print("要收敛，需要 |1 - 2*lr| < 1，因此 0 < lr < 1")
print("lr=0.5 时，一步就到最优点；lr=1.0 时等幅振荡；lr>1 会发散。")

# 图1：不同学习率的 loss 曲线
fig, ax = plt.subplots(figsize=(9, 5.5))
for lr in learning_rates:
    hist = summary_1d[lr]
    steps = [h[0] for h in hist]
    losses = [h[2] for h in hist]
    # 发散曲线可能非常大，使用对数尺度。
    ax.plot(steps, np.maximum(losses, 1e-12), marker="o", markersize=2.5, label=f"lr={lr}")
ax.set_yscale("log")
ax.set_xlabel("Step")
ax.set_ylabel("Loss (log scale)")
ax.set_title("一维二次函数：不同 Learning Rate 的收敛 / 振荡 / 发散")
ax.grid(True, alpha=0.25)
ax.legend(ncol=2)
fig.tight_layout()
fig.savefig(FIG_DIR / "lesson36_lr_1d_comparison.png", dpi=180)
plt.close(fig)

# 图2：参数 w 的轨迹
fig, ax = plt.subplots(figsize=(9, 5.5))
for lr in [0.01, 0.1, 0.5, 0.9, 1.0]:
    hist = summary_1d[lr]
    steps = [h[0] for h in hist]
    ws = [h[1] for h in hist]
    ax.plot(steps, ws, marker="o", markersize=2.5, label=f"lr={lr}")
ax.axhline(3.0, linestyle="--", linewidth=1.2, label="最优 w=3")
ax.set_xlabel("Step")
ax.set_ylabel("w")
ax.set_title("Learning Rate 改变参数跨越最优点的方式")
ax.grid(True, alpha=0.25)
ax.legend(ncol=2)
fig.tight_layout()
fig.savefig(FIG_DIR / "lesson36_parameter_trajectory.png", dpi=180)
plt.close(fig)


def anisotropic_loss(w):
    # L = 0.5*(w1^2 + 20*w2^2)
    return 0.5 * (w[0] ** 2 + 20.0 * w[1] ** 2)


def anisotropic_grad(w):
    return np.array([w[0], 20.0 * w[1]], dtype=np.float64)


def run_2d(lr: float, steps: int = 40, w0=(4.0, 1.5)):
    w = np.array(w0, dtype=np.float64)
    path = [w.copy()]
    losses = [anisotropic_loss(w)]
    for _ in range(steps):
        w = w - lr * anisotropic_grad(w)
        path.append(w.copy())
        losses.append(anisotropic_loss(w))
    return np.array(path), np.array(losses)


section("实验2：不同曲率方向——为什么陡峭方向会限制全局学习率")
# 对 w2 方向：更新因子 = 1 - 20*lr，稳定要求 |1-20lr|<1 => lr<0.1
for lr in [0.02, 0.08, 0.095, 0.105]:
    path, losses = run_2d(lr, steps=30)
    print(
        f"lr={lr:.3f}: final w=({path[-1,0]: .6f}, {path[-1,1]: .6f}), "
        f"final loss={losses[-1]: .6e}, max loss={np.max(losses): .6e}"
    )
print("理论：w2 方向曲率为20，稳定要求 lr < 2/20 = 0.1。")

# 图3：等高线 + 两条路径
x1 = np.linspace(-4.5, 4.5, 300)
x2 = np.linspace(-2.0, 2.0, 300)
X1, X2 = np.meshgrid(x1, x2)
Z = 0.5 * (X1 ** 2 + 20.0 * X2 ** 2)
fig, ax = plt.subplots(figsize=(8.5, 6.2))
levels = np.geomspace(0.05, 80, 18)
ax.contour(X1, X2, Z, levels=levels, linewidths=0.8)
for lr in [0.02, 0.08, 0.105]:
    path, _ = run_2d(lr, steps=30)
    # 对发散轨迹只画有限范围内的点，避免图完全失真。
    finite = np.isfinite(path).all(axis=1)
    path = path[finite]
    ax.plot(path[:, 0], path[:, 1], marker="o", markersize=2.5, label=f"lr={lr}")
ax.scatter([0], [0], marker="*", s=120, label="最优点")
ax.set_xlim(-4.5, 4.5)
ax.set_ylim(-2.0, 2.0)
ax.set_xlabel("w1（较平缓方向）")
ax.set_ylabel("w2（较陡方向）")
ax.set_title("同一个 Learning Rate 必须兼顾不同曲率方向")
ax.legend()
ax.grid(True, alpha=0.15)
fig.tight_layout()
fig.savefig(FIG_DIR / "lesson36_anisotropic_landscape.png", dpi=180)
plt.close(fig)

# ----------------------- PyTorch regression -----------------------
section("实验3：PyTorch 同初始化对照——学习率怎样改变真实训练轨迹")
torch.manual_seed(20261002)
np.random.seed(20261002)

# 简单回归数据 y = 2x + 1 + 小噪声
x = torch.linspace(-2, 2, 128).reshape(-1, 1)
noise = 0.08 * torch.sin(torch.arange(128, dtype=torch.float32).reshape(-1, 1))
y = 2.0 * x + 1.0 + noise

# 固定同一份初始化，确保比较只改变 lr。
base = nn.Linear(1, 1)
base_state = {k: v.detach().clone() for k, v in base.state_dict().items()}
print(f"共同初始化：weight={base.weight.item(): .6f}, bias={base.bias.item(): .6f}")


def train_linear(lr: float, steps: int = 120):
    model = nn.Linear(1, 1)
    model.load_state_dict(base_state)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    losses = []
    weights = []
    biases = []
    update_ratios = []
    for _ in range(steps):
        opt.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        # 在 step 前记录“这次 SGD 更新量 / 参数量级”。
        with torch.no_grad():
            grad_norm = torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters()))
            param_norm = torch.sqrt(sum((p ** 2).sum() for p in model.parameters()))
            update_norm = lr * grad_norm
            ratio = (update_norm / (param_norm + 1e-12)).item()
        losses.append(loss.item())
        weights.append(model.weight.item())
        biases.append(model.bias.item())
        update_ratios.append(ratio)
        opt.step()
    return {
        "losses": np.array(losses),
        "weights": np.array(weights),
        "biases": np.array(biases),
        "update_ratios": np.array(update_ratios),
        "final_weight": model.weight.item(),
        "final_bias": model.bias.item(),
    }

pt_lrs = [1e-4, 1e-2, 1e-1, 8e-1]
pt_results = {}
for lr in pt_lrs:
    result = train_linear(lr)
    pt_results[lr] = result
    print(
        f"lr={lr:g}: initial loss={result['losses'][0]:.6f}, "
        f"final loss={result['losses'][-1]:.6e}, "
        f"w={result['final_weight']:.6f}, b={result['final_bias']:.6f}, "
        f"first update/param ratio={result['update_ratios'][0]:.6f}"
    )

fig, ax = plt.subplots(figsize=(9, 5.5))
for lr in pt_lrs:
    ax.plot(np.maximum(pt_results[lr]["losses"], 1e-12), label=f"lr={lr:g}")
ax.set_yscale("log")
ax.set_xlabel("Optimizer Step")
ax.set_ylabel("MSE Loss (log scale)")
ax.set_title("同一初始化、同一数据：Learning Rate 改变训练轨迹")
ax.grid(True, alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "lesson36_pytorch_lr_comparison.png", dpi=180)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5.5))
for lr in [1e-4, 1e-2, 1e-1]:
    ax.plot(np.maximum(pt_results[lr]["update_ratios"], 1e-12), label=f"lr={lr:g}")
ax.set_yscale("log")
ax.set_xlabel("Optimizer Step")
ax.set_ylabel("||Δθ|| / ||θ|| 近似值")
ax.set_title("更新量相对参数量级：Learning Rate 的另一种诊断视角")
ax.grid(True, alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "lesson36_update_to_parameter_ratio.png", dpi=180)
plt.close(fig)

# ----------------------- SGD vs Adam -----------------------
section("实验4：相同 nominal LR 在 SGD 与 Adam 中并不代表相同实际步长")
# 使用一个简单的二维参数直接优化，让更新更透明。
start = torch.tensor([5.0, -3.0], dtype=torch.float32)


def simple_loss(theta):
    return (theta[0] - 1.0) ** 2 + 10.0 * (theta[1] + 1.0) ** 2


def one_optimizer_step(kind: str, lr: float):
    theta = nn.Parameter(start.clone())
    if kind == "SGD":
        opt = torch.optim.SGD([theta], lr=lr)
    elif kind == "Adam":
        opt = torch.optim.Adam([theta], lr=lr, betas=(0.9, 0.999), eps=1e-8)
    else:
        raise ValueError(kind)
    opt.zero_grad()
    loss = simple_loss(theta)
    loss.backward()
    grad = theta.grad.detach().clone()
    before = theta.detach().clone()
    opt.step()
    after = theta.detach().clone()
    delta = after - before
    return loss.item(), grad.numpy(), delta.numpy()

for lr in [0.01, 0.1]:
    s_loss, s_grad, s_delta = one_optimizer_step("SGD", lr)
    a_loss, a_grad, a_delta = one_optimizer_step("Adam", lr)
    print(f"lr={lr:g}")
    print(f"  shared gradient = {s_grad}")
    print(f"  SGD  Δtheta     = {s_delta}")
    print(f"  Adam Δtheta     = {a_delta}")

# 图5：相同lr下SGD与Adam第一步向量
lr_demo = 0.1
_, grad, s_delta = one_optimizer_step("SGD", lr_demo)
_, _, a_delta = one_optimizer_step("Adam", lr_demo)
fig, ax = plt.subplots(figsize=(7.5, 6.0))
ax.scatter([start[0].item()], [start[1].item()], s=80, label="起点")
ax.arrow(start[0].item(), start[1].item(), s_delta[0], s_delta[1],
         head_width=0.08, length_includes_head=True, label="SGD update")
ax.arrow(start[0].item(), start[1].item(), a_delta[0], a_delta[1],
         head_width=0.08, length_includes_head=True, label="Adam update")
ax.set_xlabel("θ1")
ax.set_ylabel("θ2")
ax.set_title("相同 nominal LR，不同 Optimizer 的实际第一步不同")
ax.grid(True, alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "lesson36_sgd_vs_adam_step.png", dpi=180)
plt.close(fig)

# ----------------------- scheduler / warmup illustration -----------------------
section("实验5：Warmup + Cosine Decay 只是改变每个 Step 使用的 LR")
peak_lr = 3e-4
warmup_steps = 20
max_steps = 200
schedule = []
for step in range(max_steps):
    if step < warmup_steps:
        lr = peak_lr * (step + 1) / warmup_steps
    else:
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        lr = peak_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    schedule.append(lr)
print(f"peak_lr={peak_lr}, warmup_steps={warmup_steps}, max_steps={max_steps}")
print(f"step 1 lr={schedule[0]:.8f}")
print(f"step {warmup_steps} lr={schedule[warmup_steps-1]:.8f}")
print(f"step {max_steps} lr={schedule[-1]:.8f}")

fig, ax = plt.subplots(figsize=(9, 5.0))
ax.plot(np.arange(1, max_steps + 1), schedule)
ax.axvline(warmup_steps, linestyle="--", linewidth=1.2, label="warmup end")
ax.set_xlabel("Optimizer Step")
ax.set_ylabel("Learning Rate")
ax.set_title("Warmup + Cosine Decay 示例")
ax.grid(True, alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "lesson36_warmup_cosine_schedule.png", dpi=180)
plt.close(fig)

print("\n实验完成。图像已保存到：", FIG_DIR)
