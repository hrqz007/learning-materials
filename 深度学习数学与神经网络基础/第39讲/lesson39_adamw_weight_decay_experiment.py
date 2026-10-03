# 第39讲实验：AdamW 与 Weight Decay
# 目标：用可复现实验理解 L2 regularization、coupled weight decay 与 decoupled AdamW 的区别。

import math
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ------------------------------
# 0. 基本设置
# ------------------------------
torch.set_printoptions(precision=8, sci_mode=False)
np.set_printoptions(precision=8, suppress=True)
BASE = Path(__file__).resolve().parent
FIG = BASE / "figures"
FIG.mkdir(exist_ok=True)
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
_cn = font_manager.FontProperties(fname=FONT_PATH)
plt.rcParams["font.family"] = _cn.get_name()
plt.rcParams["axes.unicode_minus"] = False


def title(text):
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


# ------------------------------
# 实验1：SGD 下 L2 penalty 与 decoupled shrink 等价
# ------------------------------
title("实验1：普通 SGD 下，L2 penalty 与 weight decay 的一步更新完全等价")
w = np.array([3.0, -2.0], dtype=np.float64)
g_data = np.array([0.5, -1.5], dtype=np.float64)
lr = 0.1
wd = 0.2

# 写进 loss 的 L2 penalty: (wd/2)||w||^2，其梯度为 wd*w
w_l2 = w - lr * (g_data + wd * w)
# 先按数据梯度更新，再显式做比例衰减；代数上等价于同一步公式
w_decay = (1.0 - lr * wd) * w - lr * g_data

print("初始 w               =", w)
print("data gradient        =", g_data)
print("lr, weight_decay     =", lr, wd)
print("SGD + L2 penalty     =", w_l2)
print("SGD decoupled form   =", w_decay)
print("最大绝对差异          =", float(np.max(np.abs(w_l2 - w_decay))))


# ------------------------------
# 实验2：Adam 中 coupled L2 与 AdamW 的第一步差异
# ------------------------------
title("实验2：data gradient=0 时，Adam(coupled) 与 AdamW(decoupled) 的第一步")
init = torch.tensor([10.0, 1.0, 0.1], dtype=torch.float64)
lr2 = 0.01
wd2 = 0.1

# Adam：weight_decay 默认是 coupled L2（本环境 PyTorch 2.10，decoupled_weight_decay=False）
p_adam = torch.nn.Parameter(init.clone())
opt_adam = torch.optim.Adam([p_adam], lr=lr2, weight_decay=wd2, betas=(0.9, 0.999), eps=1e-8)
p_adam.grad = torch.zeros_like(p_adam)  # data gradient 明确设为 0
opt_adam.step()

# AdamW：decoupled weight decay
p_adamw = torch.nn.Parameter(init.clone())
opt_adamw = torch.optim.AdamW([p_adamw], lr=lr2, weight_decay=wd2, betas=(0.9, 0.999), eps=1e-8)
p_adamw.grad = torch.zeros_like(p_adamw)
opt_adamw.step()

print("初始参数             =", init)
print("Adam  一步后          =", p_adam.detach())
print("AdamW 一步后          =", p_adamw.detach())
print("Adam  参数变化         =", (p_adam.detach() - init))
print("AdamW 参数变化         =", (p_adamw.detach() - init))
print("AdamW 理论比例因子     =", 1.0 - lr2 * wd2)
print("AdamW 理论结果         =", init * (1.0 - lr2 * wd2))

# 读取 optimizer state，观察 coupled Adam 的 m/v 被 weight decay 梯度污染，而 AdamW 没有
state_adam = opt_adam.state[p_adam]
state_adamw = opt_adamw.state[p_adamw]
print("Adam  exp_avg(m)       =", state_adam["exp_avg"])
print("Adam  exp_avg_sq(v)    =", state_adam["exp_avg_sq"])
print("AdamW exp_avg(m)       =", state_adamw["exp_avg"])
print("AdamW exp_avg_sq(v)    =", state_adamw["exp_avg_sq"])

# 图：第一步变化
labels = ["w=10", "w=1", "w=0.1"]
x = np.arange(len(labels))
width = 0.36
fig, ax = plt.subplots(figsize=(7.2, 4.6))
adam_change = np.abs((p_adam.detach() - init).numpy())
adamw_change = np.abs((p_adamw.detach() - init).numpy())
ax.bar(x - width/2, adam_change, width, label="Adam: coupled L2")
ax.bar(x + width/2, adamw_change, width, label="AdamW: decoupled decay")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("第一步参数变化绝对值")
ax.set_title("data gradient=0 时：coupled Adam 与 AdamW 的第一步")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "lesson39_first_step_comparison.png", dpi=180)
plt.close(fig)


# ------------------------------
# 实验3：多步 zero-data-gradient shrink trajectory
# ------------------------------
title("实验3：data gradient=0，连续 50 Step 的参数衰减轨迹")
steps = 50
pa = torch.nn.Parameter(init.clone())
pw = torch.nn.Parameter(init.clone())
oa = torch.optim.Adam([pa], lr=lr2, weight_decay=wd2)
ow = torch.optim.AdamW([pw], lr=lr2, weight_decay=wd2)
traj_a = [pa.detach().clone().numpy()]
traj_w = [pw.detach().clone().numpy()]
for _ in range(steps):
    pa.grad = torch.zeros_like(pa)
    pw.grad = torch.zeros_like(pw)
    oa.step()
    ow.step()
    traj_a.append(pa.detach().clone().numpy())
    traj_w.append(pw.detach().clone().numpy())
traj_a = np.stack(traj_a)
traj_w = np.stack(traj_w)
print("50 Step 后 Adam       =", pa.detach())
print("50 Step 后 AdamW      =", pw.detach())
print("AdamW 理论 w=10       =", 10.0 * ((1-lr2*wd2) ** steps))

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
for j, lab in enumerate(labels):
    axes[0].plot(np.arange(steps+1), traj_a[:, j], label=lab)
    axes[1].plot(np.arange(steps+1), traj_w[:, j], label=lab)
axes[0].set_title("Adam + coupled L2")
axes[1].set_title("AdamW + decoupled weight decay")
for ax in axes:
    ax.set_xlabel("Optimizer Step")
    ax.set_ylabel("参数值")
    ax.legend()
fig.suptitle("data gradient=0 时的纯衰减轨迹")
fig.tight_layout()
fig.savefig(FIG / "lesson39_zero_grad_decay_trajectories.png", dpi=180)
plt.close(fig)


# ------------------------------
# 实验4：显式 L2 penalty 与 Adam(weight_decay) 对齐
# ------------------------------
title("实验4：显式 L2 penalty 与 Adam(weight_decay) 的 coupled 语义对齐")
torch.manual_seed(7)
start = torch.tensor([1.5, -0.7], dtype=torch.float64)
data_grad = torch.tensor([0.3, -0.2], dtype=torch.float64)
lam = 0.05
lr4 = 0.01

# A: Adam weight_decay=lam，直接提供 data gradient
p_a = torch.nn.Parameter(start.clone())
o_a = torch.optim.Adam([p_a], lr=lr4, weight_decay=lam)
p_a.grad = data_grad.clone()
o_a.step()

# B: Adam weight_decay=0，但人为把 (lam/2)||w||^2 的梯度 lam*w 加进去
p_b = torch.nn.Parameter(start.clone())
o_b = torch.optim.Adam([p_b], lr=lr4, weight_decay=0.0)
p_b.grad = data_grad + lam * start
o_b.step()

print("Adam(weight_decay=lam) =", p_a.detach())
print("Adam + 显式 L2 梯度     =", p_b.detach())
print("最大绝对差异            =", float(torch.max(torch.abs(p_a.detach()-p_b.detach()))))


# ------------------------------
# 实验5：AdamW 与 Adam(decoupled_weight_decay=True) 对齐（当前 PyTorch 版本）
# ------------------------------
title("实验5：当前 PyTorch 中 AdamW 与 Adam(decoupled_weight_decay=True) 对齐")
start5 = torch.tensor([2.0, -1.0], dtype=torch.float64)
g5 = torch.tensor([0.4, -0.8], dtype=torch.float64)
pa5 = torch.nn.Parameter(start5.clone())
pb5 = torch.nn.Parameter(start5.clone())
oa5 = torch.optim.AdamW([pa5], lr=0.003, weight_decay=0.07)
ob5 = torch.optim.Adam([pb5], lr=0.003, weight_decay=0.07, decoupled_weight_decay=True)
for step in range(1, 9):
    # 为了验证 optimizer 公式，使用同一人工 gradient 序列
    grad = g5 * (1.0 + 0.1 * math.sin(step))
    pa5.grad = grad.clone()
    pb5.grad = grad.clone()
    oa5.step(); ob5.step()
    oa5.zero_grad(); ob5.zero_grad()
print("AdamW                         =", pa5.detach())
print("Adam(decoupled_weight_decay) =", pb5.detach())
print("最大绝对差异                  =", float(torch.max(torch.abs(pa5.detach()-pb5.detach()))))


# ------------------------------
# 实验6：学习率 schedule 会改变累计 weight decay 强度
# ------------------------------
title("实验6：相同 weight_decay，不同 LR schedule 会产生不同累计 shrink")
lam6 = 0.1
T = 100
lr_const = 0.01
# 不考虑 data gradient，只计算 AdamW 中显式的乘法 decay 因子
const_factors = np.full(T, 1.0 - lr_const * lam6)
cos_lrs = np.array([lr_const * 0.5 * (1 + math.cos(math.pi * t / (T-1))) for t in range(T)])
cos_factors = 1.0 - cos_lrs * lam6
w_const = [1.0]
w_cos = [1.0]
for f1, f2 in zip(const_factors, cos_factors):
    w_const.append(w_const[-1] * f1)
    w_cos.append(w_cos[-1] * f2)
print("100 Step constant LR 后纯 decay 参数 =", w_const[-1])
print("100 Step cosine LR 后纯 decay 参数   =", w_cos[-1])

fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.plot(np.arange(T+1), w_const, label="constant lr=0.01")
ax.plot(np.arange(T+1), w_cos, label="cosine lr")
ax.set_xlabel("Optimizer Step")
ax.set_ylabel("只考虑 AdamW decay 后的参数比例")
ax.set_title("AdamW 中 LR schedule 会改变累计 weight decay")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "lesson39_lr_schedule_decay.png", dpi=180)
plt.close(fig)


# ------------------------------
# 实验7：一个小回归任务，观察不同 regularization 语义的训练轨迹
# 注意：仅做机制演示，不据此声称谁的泛化一定更好。
# ------------------------------
title("实验7：小回归任务中的 No WD / Adam coupled / AdamW 对照（机制演示）")
torch.manual_seed(123)
x = torch.linspace(-2, 2, 96, dtype=torch.float32).unsqueeze(1)
y = 3.0 * x - 1.0 + 0.15 * torch.randn_like(x)


def train(kind, steps=300, lr=0.03, wd=0.08):
    torch.manual_seed(2026)
    model = torch.nn.Sequential(torch.nn.Linear(1, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1))
    if kind == "no_wd":
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    elif kind == "adam_l2":
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif kind == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise ValueError(kind)
    losses, norms = [], []
    for _ in range(steps):
        opt.zero_grad()
        pred = model(x)
        loss = torch.mean((pred-y)**2)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
        with torch.no_grad():
            norm = math.sqrt(sum(float((p**2).sum()) for p in model.parameters()))
        norms.append(norm)
    return np.array(losses), np.array(norms)

res = {k: train(k) for k in ["no_wd", "adam_l2", "adamw"]}
for k, (losses, norms) in res.items():
    print(f"{k:8s} final data loss={losses[-1]:.8f}, final parameter norm={norms[-1]:.6f}")

fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
for k, (losses, norms) in res.items():
    axes[0].plot(losses, label=k)
    axes[1].plot(norms, label=k)
axes[0].set_yscale("log")
axes[0].set_title("Data MSE（仅机制演示）")
axes[0].set_xlabel("Optimizer Step")
axes[0].set_ylabel("MSE")
axes[1].set_title("Parameter L2 norm")
axes[1].set_xlabel("Optimizer Step")
axes[1].set_ylabel("参数范数")
for ax in axes:
    ax.legend()
fig.tight_layout()
fig.savefig(FIG / "lesson39_training_mechanism_comparison.png", dpi=180)
plt.close(fig)

print("\n实验完成。图像保存在：", FIG)
