# -*- coding: utf-8 -*-
"""
第34讲实验：Full-Batch GD、SGD 与 Mini-Batch SGD
目标：
1. 比较不同 Batch Size 下 Gradient Estimate 的噪声；
2. 比较相同数据集上不同 Batch Size 的训练轨迹；
3. 验证 DataLoader 决定 Batch，而 torch.optim.SGD 负责参数更新；
4. 验证 Gradient Accumulation 与较大 Effective Batch 的近似/等价关系。
"""

from pathlib import Path
import math
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# -----------------------------
# 0. 固定随机性与输出目录
# -----------------------------
torch.manual_seed(34)
np.random.seed(34)

BASE = Path(__file__).resolve().parent
FIG_DIR = BASE / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_TXT = BASE / "第34讲_实验实际输出.txt"

# Matplotlib 中文字体设置
plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

lines = []
def log(text=""):
    print(text)
    lines.append(str(text))

# -----------------------------
# 1. 构造一个带噪声的线性回归数据集
# -----------------------------
N = 64
x = torch.linspace(-2.0, 2.0, N).reshape(-1, 1)
noise = 0.25 * torch.randn(N, 1)
y = 3.0 * x - 2.0 + noise

dataset = TensorDataset(x, y)

log("=== 实验1：数据集与三种 Gradient 计算方式 ===")
log(f"N = {N}")
log(f"x.shape = {tuple(x.shape)}")
log(f"y.shape = {tuple(y.shape)}")
log("真实生成规律：y = 3x - 2 + noise")
log()

# -----------------------------
# 2. 工具函数：在给定样本上计算 w,b 的 MSE Gradient
# 模型：pred = x*w + b
# -----------------------------
def grad_on_batch(xb, yb, w_value=0.0, b_value=0.0):
    # 创建需要求导的标量参数
    w = torch.tensor(float(w_value), requires_grad=True)
    b = torch.tensor(float(b_value), requires_grad=True)
    # Forward
    pred = xb.squeeze(-1) * w + b
    # 当前 Batch 的 mean MSE
    loss = ((pred - yb.squeeze(-1)) ** 2).mean()
    # Backward
    loss.backward()
    # 返回 Loss 和两个梯度
    return float(loss.item()), float(w.grad.item()), float(b.grad.item())

# 全数据 Gradient：Full-Batch Gradient
full_loss, full_gw, full_gb = grad_on_batch(x, y)
full_grad = np.array([full_gw, full_gb])
log(f"Full-Batch Loss = {full_loss:.6f}")
log(f"Full-Batch Gradient = [dw={full_gw:.6f}, db={full_gb:.6f}]")

# 单样本 Gradient：真正 batch_size=1 的随机梯度候选
single_grads = []
for i in range(N):
    _, gw, gb = grad_on_batch(x[i:i+1], y[i:i+1])
    single_grads.append([gw, gb])
single_grads = np.asarray(single_grads)
log(f"Single-sample gradients 数量 = {len(single_grads)}")
log(f"Single-sample dw 标准差 = {single_grads[:,0].std(ddof=1):.6f}")
log(f"Single-sample db 标准差 = {single_grads[:,1].std(ddof=1):.6f}")
log()

# -----------------------------
# 3. 重复随机抽 Mini-Batch：观察 Gradient Estimate 噪声随 B 变化
# -----------------------------
log("=== 实验2：Batch Size 与 Gradient Noise ===")
rng = np.random.default_rng(3401)
batch_sizes = [1, 2, 4, 8, 16, 32, 64]
num_draws = 500
stats = []
all_estimates = {}

for B in batch_sizes:
    estimates = []
    if B == N:
        # Full batch 每次都完全相同
        estimates = np.repeat(full_grad[None, :], num_draws, axis=0)
    else:
        for _ in range(num_draws):
            # 每次在 N 个样本中无放回随机抽 B 个
            idx = rng.choice(N, size=B, replace=False)
            _, gw, gb = grad_on_batch(x[idx], y[idx])
            estimates.append([gw, gb])
        estimates = np.asarray(estimates)
    all_estimates[B] = estimates
    mean_grad = estimates.mean(axis=0)
    std_grad = estimates.std(axis=0, ddof=1) if B < N else np.zeros(2)
    mean_l2_error = np.linalg.norm(estimates - full_grad[None, :], axis=1).mean()
    stats.append((B, mean_grad[0], mean_grad[1], std_grad[0], std_grad[1], mean_l2_error))
    log(
        f"B={B:2d} | mean_grad=[{mean_grad[0]: .4f}, {mean_grad[1]: .4f}] "
        f"| std=[{std_grad[0]:.4f}, {std_grad[1]:.4f}] "
        f"| mean ||g_batch-g_full||={mean_l2_error:.4f}"
    )
log()

# 图1：梯度空间中的随机估计
fig, ax = plt.subplots(figsize=(8, 6))
for B in [1, 8, 32]:
    est = all_estimates[B][:120]
    ax.scatter(est[:, 0], est[:, 1], s=18, alpha=0.45, label=f"Batch={B}")
ax.scatter([full_gw], [full_gb], s=130, marker="*", label="Full-Batch Gradient")
ax.set_xlabel("dL/dw")
ax.set_ylabel("dL/db")
ax.set_title("不同 Batch Size 的 Gradient Estimate 分布")
ax.legend()
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(FIG_DIR / "lesson34_gradient_estimates_scatter.png", dpi=180)
plt.close(fig)

# 图2：Gradient Noise 随 Batch Size 变化
stats_arr = np.asarray(stats, dtype=float)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(stats_arr[:,0], stats_arr[:,3], marker="o", label="std(dL/dw)")
ax.plot(stats_arr[:,0], stats_arr[:,4], marker="s", label="std(dL/db)")
ax.set_xscale("log", base=2)
ax.set_xlabel("Batch Size")
ax.set_ylabel("Gradient Estimate 标准差")
ax.set_title("Batch 越大，Gradient Noise 通常越小")
ax.legend()
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(FIG_DIR / "lesson34_gradient_noise_vs_batch.png", dpi=180)
plt.close(fig)

# -----------------------------
# 4. 比较不同 Batch Size 的训练轨迹
# 为避免“初始化不同”造成混淆，所有实验使用相同初始参数。
# -----------------------------
log("=== 实验3：Full-Batch / SGD / Mini-Batch 的训练轨迹 ===")
initial_w = torch.tensor([[-0.8]], dtype=torch.float32)
initial_b = torch.tensor([0.5], dtype=torch.float32)


def train_with_batch_size(batch_size, epochs=30, lr=0.05):
    # 每次创建结构相同的新模型
    model = nn.Linear(1, 1)
    # 固定完全相同的初始化
    with torch.no_grad():
        model.weight.copy_(initial_w)
        model.bias.copy_(initial_b)
    # 注意：SGD Optimizer 不知道 batch_size；batch_size 由 DataLoader 决定
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss(reduction="mean")
    # 为每种 B 使用固定随机种子，保证实验可复现
    generator = torch.Generator().manual_seed(3400 + int(batch_size))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)

    step_losses = []
    epoch_losses = []
    w_history = []
    b_history = []
    total_steps = 0

    for epoch in range(epochs):
        sum_loss = 0.0
        count = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

            bs = xb.shape[0]
            # 按样本数加权累计 epoch loss
            sum_loss += float(loss.item()) * bs
            count += bs
            step_losses.append(float(loss.item()))
            total_steps += 1
        epoch_losses.append(sum_loss / count)
        w_history.append(float(model.weight.item()))
        b_history.append(float(model.bias.item()))

    return {
        "model": model,
        "step_losses": np.asarray(step_losses),
        "epoch_losses": np.asarray(epoch_losses),
        "w_history": np.asarray(w_history),
        "b_history": np.asarray(b_history),
        "steps": total_steps,
    }

train_batch_sizes = [1, 8, 32, 64]
train_results = {}
for B in train_batch_sizes:
    res = train_with_batch_size(B)
    train_results[B] = res
    model = res["model"]
    steps_per_epoch = math.ceil(N / B)
    log(
        f"B={B:2d} | steps/epoch={steps_per_epoch:2d} | total_steps={res['steps']:4d} "
        f"| final_epoch_loss={res['epoch_losses'][-1]:.6f} "
        f"| w={model.weight.item():.4f}, b={model.bias.item():.4f}"
    )
log("注意：这里固定相同 epoch 数，因此不同 Batch Size 的 optimizer step 数不同。")
log("不能仅凭这一个玩具实验断言哪种 Batch Size 普遍最好。")
log()

# 图3：Epoch Loss
fig, ax = plt.subplots(figsize=(8, 5))
for B in train_batch_sizes:
    ax.plot(np.arange(1, len(train_results[B]["epoch_losses"])+1), train_results[B]["epoch_losses"], label=f"B={B}")
ax.set_yscale("log")
ax.set_xlabel("Epoch")
ax.set_ylabel("Mean Training MSE (log scale)")
ax.set_title("相同 Epoch 数下，不同 Batch Size 的训练曲线")
ax.legend()
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(FIG_DIR / "lesson34_training_loss_by_batch.png", dpi=180)
plt.close(fig)

# 图4：参数轨迹（每个 epoch 记录一次）
fig, ax = plt.subplots(figsize=(7, 6))
for B in train_batch_sizes:
    ax.plot(train_results[B]["w_history"], train_results[B]["b_history"], marker="o", markersize=2.5, label=f"B={B}")
ax.scatter([3.0], [-2.0], marker="*", s=150, label="生成规律 (3,-2)")
ax.set_xlabel("weight w")
ax.set_ylabel("bias b")
ax.set_title("不同 Batch Size 的参数更新轨迹（每 Epoch 记录）")
ax.legend()
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(FIG_DIR / "lesson34_parameter_trajectories.png", dpi=180)
plt.close(fig)

# 图5：每个 epoch 的 step 数
steps_per_epoch = [math.ceil(N / B) for B in batch_sizes]
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar([str(B) for B in batch_sizes], steps_per_epoch)
ax.set_xlabel("Batch Size")
ax.set_ylabel("Optimizer Steps per Epoch")
ax.set_title(f"N={N} 时 Batch Size 与每 Epoch Step 数")
fig.tight_layout()
fig.savefig(FIG_DIR / "lesson34_steps_per_epoch.png", dpi=180)
plt.close(fig)

# -----------------------------
# 5. Gradient Accumulation：4个 micro-batch = 一个 effective batch
# -----------------------------
log("=== 实验4：Gradient Accumulation 与 Effective Batch Size ===")
# 取固定的 16 个样本
xbig = x[:16]
ybig = y[:16]

# 建立两个完全相同的模型
model_big = nn.Linear(1, 1)
model_acc = nn.Linear(1, 1)
with torch.no_grad():
    model_big.weight.copy_(initial_w)
    model_big.bias.copy_(initial_b)
    model_acc.weight.copy_(initial_w)
    model_acc.bias.copy_(initial_b)

loss_fn = nn.MSELoss(reduction="mean")

# A. 一次性 Batch=16
pred_big = model_big(xbig)
loss_big = loss_fn(pred_big, ybig)
loss_big.backward()
big_gw = model_big.weight.grad.detach().clone()
big_gb = model_big.bias.grad.detach().clone()

# B. 4个 micro-batch，每个4个；loss / 4 后累积梯度
model_acc.zero_grad()
acc_steps = 4
micro = 4
for k in range(acc_steps):
    xs = xbig[k*micro:(k+1)*micro]
    ys = ybig[k*micro:(k+1)*micro]
    pred = model_acc(xs)
    loss = loss_fn(pred, ys) / acc_steps
    loss.backward()
acc_gw = model_acc.weight.grad.detach().clone()
acc_gb = model_acc.bias.grad.detach().clone()

log(f"Big Batch=16 gradient: dw={big_gw.item():.8f}, db={big_gb.item():.8f}")
log(f"4 x Micro-Batch=4 accumulated gradient: dw={acc_gw.item():.8f}, db={acc_gb.item():.8f}")
log(f"|dw difference| = {abs(big_gw.item()-acc_gw.item()):.3e}")
log(f"|db difference| = {abs(big_gb.item()-acc_gb.item()):.3e}")
log("若每个 micro-batch loss 使用 mean reduction，则通常需要除以 accumulation_steps，才能匹配大 Batch 的 mean gradient。")
log()

# 图6：大 Batch 与累积梯度比较
fig, ax = plt.subplots(figsize=(7, 5))
labels = ["dL/dw", "dL/db"]
xpos = np.arange(2)
width = 0.36
ax.bar(xpos - width/2, [big_gw.item(), big_gb.item()], width=width, label="Batch=16")
ax.bar(xpos + width/2, [acc_gw.item(), acc_gb.item()], width=width, label="4×MicroBatch=4")
ax.set_xticks(xpos)
ax.set_xticklabels(labels)
ax.set_ylabel("Gradient")
ax.set_title("Gradient Accumulation 与大 Batch 的梯度对照")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "lesson34_gradient_accumulation.png", dpi=180)
plt.close(fig)

# -----------------------------
# 6. 最终科研边界
# -----------------------------
log("=== 实验结论边界 ===")
log("1. Mini-Batch gradient 是 Full-Batch gradient 的随机估计；Batch 越小，通常波动越大。")
log("2. Batch 越大不代表训练一定更好，也不代表 wall-clock 一定更快；硬件、学习率、并行效率、数据与模型都会影响结果。")
log("3. 比较 Batch Size 时必须说明比较基准：相同 epoch、相同 optimizer steps、相同 seen samples/tokens，结论可能不同。")
log("4. torch.optim.SGD 负责更新规则；DataLoader 的 batch_size 决定每次用多少样本估计 gradient。")
log("5. Gradient Accumulation 可以提高 effective batch size，但不会把 micro-batch 的峰值激活显存 magically 变成零；它主要减少一次同时驻留的样本数。")

OUTPUT_TXT.write_text("\n".join(lines), encoding="utf-8")
