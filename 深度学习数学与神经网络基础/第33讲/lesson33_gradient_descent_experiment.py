# 第33讲实验：Gradient Descent（梯度下降）
# 目标：把“梯度 -> 负梯度方向 -> 学习率 -> 参数更新 -> Loss 变化”完整做成可观察实验。

from pathlib import Path
import sys
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import font_manager

# -----------------------------------------------------------------------------
# 绘图与输出目录
# -----------------------------------------------------------------------------
CJK_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(CJK_FONT_PATH)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=CJK_FONT_PATH).get_name()
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)


def title(text):
    """打印实验标题。"""
    print("\n" + "=" * 96)
    print(text)
    print("=" * 96)


# -----------------------------------------------------------------------------
# 实验0：环境信息
# -----------------------------------------------------------------------------
title("实验0：环境信息")
print("Python:", sys.version.split()[0])
print("NumPy:", np.__version__)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())


# -----------------------------------------------------------------------------
# 实验1：标量参数的一步 Gradient Descent
# L(w) = (w - 3)^2
# -----------------------------------------------------------------------------
title("实验1：标量参数的一步 Gradient Descent")


def scalar_loss(w):
    """标量 Loss：最低点在 w=3。"""
    return (w - 3.0) ** 2


def scalar_grad(w):
    """dL/dw = 2(w-3)。"""
    return 2.0 * (w - 3.0)


w0 = 8.0
lr = 0.1
loss0 = scalar_loss(w0)
g0 = scalar_grad(w0)
w1 = w0 - lr * g0
loss1 = scalar_loss(w1)
print(f"w0 = {w0:.4f}")
print(f"L(w0) = {loss0:.4f}")
print(f"dL/dw = {g0:.4f}")
print(f"learning rate = {lr:.4f}")
print(f"update = -lr * grad = {-lr * g0:.4f}")
print(f"w1 = {w1:.4f}")
print(f"L(w1) = {loss1:.4f}")
print("Loss 是否下降:", loss1 < loss0)

# 绘制一次更新
xs = np.linspace(-1, 9, 500)
ys = scalar_loss(xs)
plt.figure(figsize=(8, 5))
plt.plot(xs, ys, label=r"$L(w)=(w-3)^2$")
plt.scatter([w0, w1], [loss0, loss1], s=70)
plt.annotate("更新前", (w0, loss0), xytext=(7.0, 30), arrowprops={"arrowstyle": "->"})
plt.annotate("更新后", (w1, loss1), xytext=(5.0, 23), arrowprops={"arrowstyle": "->"})
plt.axvline(3, linestyle="--", linewidth=1, label="minimum w=3")
plt.xlabel("参数 w")
plt.ylabel("Loss")
plt.title("一次 Gradient Descent 更新")
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "lesson33_one_step_update.png", dpi=180)
plt.close()


# -----------------------------------------------------------------------------
# 实验2：连续多步更新，看参数如何走向最低点
# -----------------------------------------------------------------------------
title("实验2：连续多步 Gradient Descent")

w = 8.0
lr = 0.1
steps = 25
w_history = [w]
loss_history = [scalar_loss(w)]
grad_history = [scalar_grad(w)]
for step in range(steps):
    grad = scalar_grad(w)
    w = w - lr * grad
    w_history.append(w)
    loss_history.append(scalar_loss(w))
    grad_history.append(scalar_grad(w))
    if step in [0, 1, 2, 4, 9, 24]:
        print(
            f"after step {step+1:2d}: w={w:.8f}, "
            f"loss={scalar_loss(w):.10f}, grad={scalar_grad(w):.8f}"
        )
print(f"最终 w = {w:.10f}")
print(f"最终 Loss = {scalar_loss(w):.12f}")

plt.figure(figsize=(8, 5))
plt.plot(range(len(loss_history)), loss_history, marker="o", markersize=3)
plt.yscale("log")
plt.xlabel("Step")
plt.ylabel("Loss（log scale）")
plt.title("Gradient Descent：Loss 随 Step 下降")
plt.tight_layout()
plt.savefig(FIG_DIR / "lesson33_loss_over_steps.png", dpi=180)
plt.close()


# -----------------------------------------------------------------------------
# 实验3：Learning Rate 太小、合适、较大、过大的对照
# 使用同一 w0，确保公平比较
# -----------------------------------------------------------------------------
title("实验3：Learning Rate 对收敛轨迹的影响")

learning_rates = [0.05, 0.2, 0.9, 1.1]
comparison = {}
for eta in learning_rates:
    w = 8.0
    ws = [w]
    losses = [scalar_loss(w)]
    for _ in range(20):
        w = w - eta * scalar_grad(w)
        ws.append(w)
        losses.append(scalar_loss(w))
    comparison[eta] = (ws, losses)
    print(
        f"lr={eta:>4}: final_w={w: .6f}, "
        f"final_loss={scalar_loss(w):.6f}, max_loss={max(losses):.6f}"
    )

plt.figure(figsize=(8, 5))
for eta, (_, losses) in comparison.items():
    plt.plot(range(len(losses)), losses, marker="o", markersize=2.5, label=f"lr={eta}")
plt.yscale("log")
plt.xlabel("Step")
plt.ylabel("Loss（log scale）")
plt.title("同一初始点下，不同 Learning Rate 的训练轨迹")
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "lesson33_learning_rate_comparison.png", dpi=180)
plt.close()


# -----------------------------------------------------------------------------
# 实验4：二维 Gradient Descent —— 向量参数一起更新
# L(w1,w2)=(w1-2)^2 + 4(w2+1)^2
# -----------------------------------------------------------------------------
title("实验4：二维参数空间中的 Gradient Descent")


def loss2d(theta):
    w1, w2 = theta
    return (w1 - 2.0) ** 2 + 4.0 * (w2 + 1.0) ** 2


def grad2d(theta):
    w1, w2 = theta
    return np.array([2.0 * (w1 - 2.0), 8.0 * (w2 + 1.0)], dtype=np.float64)


theta = np.array([-3.0, 3.0], dtype=np.float64)
lr2 = 0.12
traj = [theta.copy()]
for step in range(30):
    theta = theta - lr2 * grad2d(theta)
    traj.append(theta.copy())
    if step in [0, 1, 2, 4, 9, 29]:
        print(
            f"after step {step+1:2d}: theta={theta.tolist()}, "
            f"loss={loss2d(theta):.10f}, grad_norm={np.linalg.norm(grad2d(theta)):.8f}"
        )
print("二维最终参数 =", theta.tolist())
print("二维最终 Loss =", loss2d(theta))

traj = np.array(traj)
w1_grid = np.linspace(-4, 4.5, 300)
w2_grid = np.linspace(-3.5, 4, 300)
W1, W2 = np.meshgrid(w1_grid, w2_grid)
Z = (W1 - 2.0) ** 2 + 4.0 * (W2 + 1.0) ** 2
plt.figure(figsize=(8, 6))
levels = np.geomspace(0.05, 150, 18)
plt.contour(W1, W2, Z, levels=levels)
plt.plot(traj[:, 0], traj[:, 1], marker="o", markersize=3, linewidth=1.5, label="GD trajectory")
plt.scatter([2], [-1], marker="*", s=160, label="minimum")
plt.xlabel(r"$w_1$")
plt.ylabel(r"$w_2$")
plt.title("二维 Loss Landscape 中的 Gradient Descent 轨迹")
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "lesson33_2d_descent_trajectory.png", dpi=180)
plt.close()


# -----------------------------------------------------------------------------
# 实验5：手工更新 vs torch.optim.SGD —— 同一个参数、同一个梯度
# -----------------------------------------------------------------------------
title("实验5：手工 Gradient Descent vs torch.optim.SGD")

manual_w = 8.0
manual_grad = scalar_grad(manual_w)
manual_new = manual_w - 0.1 * manual_grad

p = torch.tensor(8.0, requires_grad=True)
optimizer = torch.optim.SGD([p], lr=0.1)
loss_t = (p - 3.0) ** 2
optimizer.zero_grad()
loss_t.backward()
print("PyTorch backward 后 p.grad =", p.grad.item())
optimizer.step()
print("手工更新后的 w =", manual_new)
print("torch.optim.SGD 更新后的 p =", p.item())
print("两者绝对差 =", abs(manual_new - p.item()))


# -----------------------------------------------------------------------------
# 实验6：完整 Full-Batch Gradient Descent 学一个线性模型 y=2x+1
# 这里每个 Step 都使用整个 Dataset，用来和下一讲 Mini-Batch SGD 区分。
# -----------------------------------------------------------------------------
title("实验6：Full-Batch Gradient Descent 训练线性回归")

torch.manual_seed(33)
X = torch.linspace(-2.0, 2.0, 41).reshape(-1, 1)
y = 2.0 * X + 1.0
model = torch.nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss_fn = torch.nn.MSELoss()
full_losses = []
for step in range(80):
    optimizer.zero_grad()
    pred = model(X)              # 每个 step 使用全部 41 个样本
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()
    full_losses.append(loss.item())
    if step in [0, 1, 2, 4, 9, 19, 39, 79]:
        print(
            f"step={step+1:2d} | loss={loss.item():.10f} | "
            f"weight={model.weight.item():.8f} | bias={model.bias.item():.8f}"
        )
print("最终 weight =", model.weight.item())
print("最终 bias   =", model.bias.item())
print("最终 loss   =", full_losses[-1])

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(full_losses) + 1), full_losses)
plt.yscale("log")
plt.xlabel("Full-batch Step")
plt.ylabel("MSE Loss（log scale）")
plt.title("Full-Batch Gradient Descent 训练线性回归")
plt.tight_layout()
plt.savefig(FIG_DIR / "lesson33_full_batch_linear_regression.png", dpi=180)
plt.close()


# -----------------------------------------------------------------------------
# 实验7：局部一阶近似为什么只在“小步”附近更可靠
# L(w)=(w-3)^2，w0=8
# ΔL ≈ grad * Δw
# -----------------------------------------------------------------------------
title("实验7：局部线性近似与步长")

w0 = 8.0
g = scalar_grad(w0)
for delta_w in [-0.01, -0.1, -0.5, -2.0, -5.0]:
    predicted_delta = g * delta_w
    actual_delta = scalar_loss(w0 + delta_w) - scalar_loss(w0)
    print(
        f"delta_w={delta_w: .2f} | 一阶预测ΔL={predicted_delta: .6f} | "
        f"实际ΔL={actual_delta: .6f} | 差={actual_delta-predicted_delta: .6f}"
    )

# 画局部切线与真实 Loss
x_local = np.linspace(2.5, 8.5, 300)
tangent = scalar_loss(w0) + g * (x_local - w0)
plt.figure(figsize=(8, 5))
plt.plot(x_local, scalar_loss(x_local), label="真实 Loss")
plt.plot(x_local, tangent, linestyle="--", label="w0=8 处的一阶切线近似")
plt.scatter([w0], [scalar_loss(w0)], s=60)
plt.xlabel("w")
plt.ylabel("Loss")
plt.title("Gradient 给的是局部一阶信息：步子越大，近似越可能失真")
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "lesson33_local_linear_approximation.png", dpi=180)
plt.close()

print("\n所有实验完成。图像已保存到:", FIG_DIR)
