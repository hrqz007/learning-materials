# -*- coding: utf-8 -*-
"""
第38讲实验：Adam Optimizer

目标：
1. 手算 Adam 的 m_t、v_t、bias correction 与参数更新。
2. 观察 Adam 如何对不同梯度尺度做坐标级自适应归一化。
3. 验证 bias correction 对训练早期更新尺度的重要性。
4. 比较 SGD / Momentum / Adam 在不同曲率方向上的行为。
5. 用 PyTorch torch.optim.Adam 与手写公式逐步对齐。
6. 观察 optimizer.state_dict() 中 Adam 的 first/second moment state。
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import torch

BASE = Path(__file__).resolve().parent
FIG_DIR = BASE / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
if font_path.exists():
    font_manager.fontManager.addfont(str(font_path))
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
else:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "PingFang SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def section(title: str):
    print("\n" + "=" * 92)
    print(title)
    print("=" * 92)


def adam_step_numpy(param, grad, m, v, t, lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8, bias_correction=True):
    """手写 Adam 一步；所有输入均为 numpy array。"""
    m = beta1 * m + (1.0 - beta1) * grad
    v = beta2 * v + (1.0 - beta2) * (grad ** 2)
    if bias_correction:
        m_used = m / (1.0 - beta1 ** t)
        v_used = v / (1.0 - beta2 ** t)
    else:
        m_used = m
        v_used = v
    update = lr * m_used / (np.sqrt(v_used) + eps)
    param = param - update
    return param, m, v, update, m_used, v_used


# -----------------------------------------------------------------------------
# 实验1：单参数 5 Step 手算 Adam
# -----------------------------------------------------------------------------
section("实验1：单参数 5 Step——逐步查看 m、v、bias correction 与 update")

lr = 0.1
beta1 = 0.9
beta2 = 0.999
eps = 1e-8
grads = [2.0, 2.0, 1.0, -1.0, 0.5]
param = np.array([5.0], dtype=np.float64)
m = np.zeros_like(param)
v = np.zeros_like(param)
rows = []

print(f"初始参数 theta={param[0]:.6f}, lr={lr}, beta1={beta1}, beta2={beta2}, eps={eps}")
for t, g_scalar in enumerate(grads, start=1):
    g = np.array([g_scalar], dtype=np.float64)
    before = float(param[0])
    param, m, v, update, m_hat, v_hat = adam_step_numpy(
        param, g, m, v, t, lr=lr, beta1=beta1, beta2=beta2, eps=eps, bias_correction=True
    )
    rows.append((t, g_scalar, float(m[0]), float(v[0]), float(m_hat[0]), float(v_hat[0]), float(update[0]), float(param[0])))
    print(
        f"step={t}: g={g_scalar:+.3f}, m={m[0]:+.6f}, v={v[0]:.6f}, "
        f"m_hat={m_hat[0]:+.6f}, v_hat={v_hat[0]:.6f}, "
        f"update={update[0]:+.6f}, theta: {before:.6f}->{param[0]:.6f}"
    )

# 图：m, m_hat, sqrt(v_hat), update
arr = np.array(rows, dtype=np.float64)
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(arr[:, 0], arr[:, 2], marker="o", label="m_t")
ax.plot(arr[:, 0], arr[:, 4], marker="o", label="m_hat_t")
ax.plot(arr[:, 0], np.sqrt(arr[:, 5]), marker="o", label="sqrt(v_hat_t)")
ax.set_xlabel("Step")
ax.set_ylabel("Value")
ax.set_title("Adam 内部状态：一阶矩、bias-corrected 一阶矩与二阶尺度")
ax.grid(True, alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "lesson38_adam_internal_states.png", dpi=180)
plt.close(fig)


# -----------------------------------------------------------------------------
# 实验2：Bias correction 的必要性
# -----------------------------------------------------------------------------
section("实验2：Bias correction——零初始化为什么会让早期 moment 偏向 0")

constant_grad = 2.0
m = 0.0
v = 0.0
bias_rows = []
for t in range(1, 11):
    m = beta1 * m + (1 - beta1) * constant_grad
    v = beta2 * v + (1 - beta2) * (constant_grad ** 2)
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    ratio_raw = m / (np.sqrt(v) + eps)
    ratio_corrected = m_hat / (np.sqrt(v_hat) + eps)
    bias_rows.append((t, m, m_hat, v, v_hat, ratio_raw, ratio_corrected))
    print(
        f"t={t:02d}: m={m:.6f}, m_hat={m_hat:.6f}, v={v:.6f}, v_hat={v_hat:.6f}, "
        f"raw_ratio={ratio_raw:.6f}, corrected_ratio={ratio_corrected:.6f}"
    )

bias_arr = np.array(bias_rows, dtype=np.float64)
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(bias_arr[:, 0], bias_arr[:, 1], marker="o", label="m_t (uncorrected)")
ax.plot(bias_arr[:, 0], bias_arr[:, 2], marker="o", label="m_hat_t")
ax.axhline(constant_grad, linestyle="--", label="真实常量 Gradient = 2")
ax.set_xlabel("Step")
ax.set_ylabel("First moment")
ax.set_title("Bias correction：m_t 从 0 起步会早期偏小，m_hat_t 恢复正确尺度")
ax.grid(True, alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "lesson38_bias_correction_first_moment.png", dpi=180)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(bias_arr[:, 0], bias_arr[:, 5], marker="o", label="未校正 m/sqrt(v)")
ax.plot(bias_arr[:, 0], bias_arr[:, 6], marker="o", label="校正后 m_hat/sqrt(v_hat)")
ax.axhline(1.0, linestyle="--", label="常量正 Gradient 的理想归一化比例")
ax.set_xlabel("Step")
ax.set_ylabel("Normalized direction")
ax.set_title("不做 bias correction 会扭曲训练早期的实际更新尺度")
ax.grid(True, alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "lesson38_bias_correction_update_ratio.png", dpi=180)
plt.close(fig)


# -----------------------------------------------------------------------------
# 实验3：不同 Gradient 尺度的坐标自适应
# -----------------------------------------------------------------------------
section("实验3：Adam 的坐标级自适应——大梯度与小梯度如何被重新缩放")

initial = np.array([0.0, 0.0], dtype=np.float64)
g = np.array([100.0, 1.0], dtype=np.float64)
sgd_lr = 0.1
adam_lr = 0.1
sgd_update = sgd_lr * g
m0 = np.zeros_like(initial)
v0 = np.zeros_like(initial)
adam_param, _, _, adam_update, mhat, vhat = adam_step_numpy(
    initial.copy(), g, m0, v0, 1, lr=adam_lr, beta1=beta1, beta2=beta2, eps=eps, bias_correction=True
)
print("Gradient =", g.tolist())
print("SGD update magnitude per coordinate =", sgd_update.tolist())
print("Adam m_hat =", np.round(mhat, 8).tolist())
print("Adam v_hat =", np.round(vhat, 8).tolist())
print("Adam update magnitude per coordinate =", np.round(adam_update, 8).tolist())
print("解释：第一步、epsilon 可忽略时，Adam 对常量尺度的影响近似归一化，因此两个坐标更新尺度接近 lr。")

labels = ["大梯度坐标", "小梯度坐标"]
x = np.arange(2)
width = 0.35
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.bar(x - width / 2, np.abs(sgd_update), width, label="SGD |update|")
ax.bar(x + width / 2, np.abs(adam_update), width, label="Adam |update|")
ax.set_xticks(x, labels)
ax.set_ylabel("Absolute update")
ax.set_title("同一 nominal LR 下：SGD 与 Adam 对不同 Gradient 尺度的响应")
ax.grid(True, axis="y", alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "lesson38_coordinate_adaptation.png", dpi=180)
plt.close(fig)


# -----------------------------------------------------------------------------
# 实验4：SGD / Momentum / Adam 在不同曲率方向上的比较
# -----------------------------------------------------------------------------
section("实验4：SGD / Momentum / Adam——在不同曲率方向上的训练轨迹")

# Loss = 0.5 * (w1^2 + 100*w2^2)
def loss2(w):
    return 0.5 * (w[0] ** 2 + 100.0 * w[1] ** 2)


def grad2(w):
    return np.array([w[0], 100.0 * w[1]], dtype=np.float64)

start = np.array([8.0, 1.5], dtype=np.float64)
steps = 150


def run_sgd(lr=0.015):
    w = start.copy(); path=[w.copy()]; losses=[loss2(w)]
    for _ in range(steps):
        w = w - lr * grad2(w)
        path.append(w.copy()); losses.append(loss2(w))
    return np.array(path), np.array(losses)


def run_momentum(lr=0.015, mu=0.9):
    w = start.copy(); buf=np.zeros_like(w); path=[w.copy()]; losses=[loss2(w)]
    for _ in range(steps):
        g = grad2(w)
        buf = mu * buf + g
        w = w - lr * buf
        path.append(w.copy()); losses.append(loss2(w))
    return np.array(path), np.array(losses)


def run_adam(lr=0.15, beta1=0.9, beta2=0.999):
    w = start.copy(); m=np.zeros_like(w); v=np.zeros_like(w); path=[w.copy()]; losses=[loss2(w)]
    for t in range(1, steps + 1):
        g = grad2(w)
        w, m, v, _, _, _ = adam_step_numpy(w, g, m, v, t, lr=lr, beta1=beta1, beta2=beta2, eps=eps, bias_correction=True)
        path.append(w.copy()); losses.append(loss2(w))
    return np.array(path), np.array(losses)

path_sgd, loss_sgd = run_sgd()
path_mom, loss_mom = run_momentum()
path_adam, loss_adam = run_adam()
print(f"起点 {start.tolist()}，初始 Loss={loss2(start):.6f}")
print(f"SGD final Loss={loss_sgd[-1]:.8f}, final w={np.round(path_sgd[-1], 6).tolist()}")
print(f"Momentum final Loss={loss_mom[-1]:.8f}, final w={np.round(path_mom[-1], 6).tolist()}")
print(f"Adam final Loss={loss_adam[-1]:.8f}, final w={np.round(path_adam[-1], 6).tolist()}")
print("注意：这里为教学演示分别选择了适合各优化器的 LR，不是公平 benchmark，也不能据此宣称某优化器普遍最好。")

w1 = np.linspace(-8.5, 8.5, 320)
w2 = np.linspace(-1.8, 1.8, 320)
W1, W2 = np.meshgrid(w1, w2)
Z = 0.5 * (W1 ** 2 + 100.0 * W2 ** 2)
fig, ax = plt.subplots(figsize=(9, 6))
levels = np.geomspace(0.05, 180, 24)
ax.contour(W1, W2, Z, levels=levels, linewidths=0.8)
ax.plot(path_sgd[:, 0], path_sgd[:, 1], marker="o", markersize=1.8, label="SGD")
ax.plot(path_mom[:, 0], path_mom[:, 1], marker="o", markersize=1.8, label="Momentum")
ax.plot(path_adam[:, 0], path_adam[:, 1], marker="o", markersize=1.8, label="Adam")
ax.scatter([0], [0], marker="*", s=120, label="minimum")
ax.set_xlabel("w1（较平缓方向）")
ax.set_ylabel("w2（高曲率方向）")
ax.set_title("不同优化器在各向异性 Loss Landscape 中的轨迹（教学示例）")
ax.grid(True, alpha=0.15)
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "lesson38_optimizer_paths.png", dpi=180)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(np.maximum(loss_sgd, 1e-14), label="SGD")
ax.plot(np.maximum(loss_mom, 1e-14), label="Momentum")
ax.plot(np.maximum(loss_adam, 1e-14), label="Adam")
ax.set_yscale("log")
ax.set_xlabel("Step")
ax.set_ylabel("Loss (log scale)")
ax.set_title("SGD / Momentum / Adam Loss 曲线（教学示例，不是公平 benchmark）")
ax.grid(True, alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "lesson38_optimizer_loss_curves.png", dpi=180)
plt.close(fig)


# -----------------------------------------------------------------------------
# 实验5：手写 Adam 与 PyTorch torch.optim.Adam 对齐
# -----------------------------------------------------------------------------
section("实验5：手写 Adam 与 torch.optim.Adam 数值逐步对齐")

torch.manual_seed(38)
initial_t = torch.tensor([2.0, -1.0], dtype=torch.float64)
manual_param = initial_t.numpy().copy()
manual_m = np.zeros(2, dtype=np.float64)
manual_v = np.zeros(2, dtype=np.float64)
param = torch.nn.Parameter(initial_t.clone())
pt_lr = 0.05
pt_beta1, pt_beta2 = 0.9, 0.999
optimizer = torch.optim.Adam([param], lr=pt_lr, betas=(pt_beta1, pt_beta2), eps=eps, weight_decay=0.0)

for t in range(1, 9):
    # L = 0.5*(p1^2 + 3*p2^2)，Gradient = [p1, 3*p2]
    manual_grad = np.array([manual_param[0], 3.0 * manual_param[1]], dtype=np.float64)
    manual_param, manual_m, manual_v, _, _, _ = adam_step_numpy(
        manual_param, manual_grad, manual_m, manual_v, t,
        lr=pt_lr, beta1=pt_beta1, beta2=pt_beta2, eps=eps, bias_correction=True
    )

    optimizer.zero_grad()
    loss = 0.5 * (param[0] ** 2 + 3.0 * param[1] ** 2)
    loss.backward()
    optimizer.step()

    diff = np.max(np.abs(manual_param - param.detach().numpy()))
    print(
        f"step={t:02d} manual={np.round(manual_param, 10).tolist()} "
        f"torch={np.round(param.detach().numpy(), 10).tolist()} max_abs_diff={diff:.3e}"
    )

state = optimizer.state[param]
print("\nPyTorch Adam state keys:", sorted([str(k) for k in state.keys()]))
print("state['step'] =", float(state["step"]))
print("exp_avg (first moment state) =", np.round(state["exp_avg"].detach().numpy(), 8).tolist())
print("exp_avg_sq (second moment state) =", np.round(state["exp_avg_sq"].detach().numpy(), 8).tolist())


# -----------------------------------------------------------------------------
# 实验6：epsilon 对极小梯度坐标的影响
# -----------------------------------------------------------------------------
section("实验6：epsilon 并不只是防止除 0——极小二阶矩时它会影响实际更新")

small_grads = np.logspace(-12, -1, 200)
updates_by_eps = {}
for eps_value in [1e-8, 1e-6, 1e-4]:
    # 第一步 bias correction 后 mhat=g, vhat=g^2
    updates = 0.001 * small_grads / (np.sqrt(small_grads ** 2) + eps_value)
    updates_by_eps[eps_value] = updates
    print(
        f"eps={eps_value:g}: g=1e-12 时 update={updates[0]:.3e}, "
        f"g=1e-4 附近 update={updates[np.argmin(np.abs(small_grads-1e-4))]:.3e}, "
        f"g=1e-1 时 update={updates[-1]:.3e}"
    )

fig, ax = plt.subplots(figsize=(9, 5.5))
for eps_value, updates in updates_by_eps.items():
    ax.plot(small_grads, updates, label=f"eps={eps_value:g}")
ax.set_xscale("log")
ax.set_xlabel("|Gradient|")
ax.set_ylabel("First-step |update| (lr=0.001)")
ax.set_title("epsilon 对极小 Gradient / second moment 区域的实际更新尺度影响")
ax.grid(True, alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "lesson38_epsilon_effect.png", dpi=180)
plt.close(fig)

print("\n实验完成。图像目录：", FIG_DIR)
