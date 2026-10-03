# 第40讲实验：Initialization、Vanishing Gradient 与 Exploding Gradient
# 运行环境：Python 3 + NumPy + PyTorch + Matplotlib

import math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 固定随机种子，保证实验可复现
torch.manual_seed(40)
np.random.seed(40)

OUT_DIR = Path(__file__).resolve().parent
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

# 尽量使用支持中文的字体；缺失时 Matplotlib 会自动回退
font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
_cjk_name = font_manager.FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc").get_name()
plt.rcParams["font.family"] = _cjk_name
plt.rcParams["axes.unicode_minus"] = False


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# -----------------------------------------------------------------------------
# 实验1：全零初始化为何会让隐藏神经元保持完全对称
# -----------------------------------------------------------------------------
section("实验1：全零初始化与隐藏神经元对称性")

# 两个输入特征，两个隐藏神经元，一个输出神经元
x = torch.tensor([[1.0, -2.0]])
y = torch.tensor([[1.0]])

# 第一层两个隐藏神经元使用完全相同（全零）的权重和偏置
W1 = torch.zeros(2, 2, requires_grad=True)
b1 = torch.zeros(2, requires_grad=True)
# 为了避免输出也完全为零导致某些路径过于退化，第二层固定为相同非零权重
W2 = torch.tensor([[1.0], [1.0]], requires_grad=True)
b2 = torch.zeros(1, requires_grad=True)

# 使用 tanh：在 0 点导数为 1，便于清楚看到对称梯度
z1 = x @ W1 + b1
h = torch.tanh(z1)
pred = h @ W2 + b2
loss = torch.mean((pred - y) ** 2)
loss.backward()

print("W1 初始值:\n", W1.detach().numpy())
print("W1.grad:\n", W1.grad.detach().numpy())
print("b1.grad:", b1.grad.detach().numpy())
print("第一层两列梯度是否完全相同:", bool(torch.allclose(W1.grad[:, 0], W1.grad[:, 1])))

# 做一步相同的 SGD 更新，观察两个隐藏神经元仍保持一样
lr = 0.1
with torch.no_grad():
    W1_new = W1 - lr * W1.grad
    b1_new = b1 - lr * b1.grad
print("更新后 W1:\n", W1_new.numpy())
print("更新后两个隐藏神经元是否仍完全对称:", bool(torch.allclose(W1_new[:, 0], W1_new[:, 1])))

fig, ax = plt.subplots(figsize=(8, 4.8))
vals = W1.grad.detach().numpy().T  # 每行表示一个隐藏神经元收到的两个输入权重梯度
xpos = np.arange(2)
width = 0.34
ax.bar(xpos - width/2, vals[0], width, label="隐藏神经元 1")
ax.bar(xpos + width/2, vals[1], width, label="隐藏神经元 2")
ax.set_xticks(xpos, ["输入权重 1", "输入权重 2"])
ax.set_ylabel("Gradient")
ax.set_title("全零初始化：对称隐藏神经元收到相同 Gradient")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "lesson40_zero_init_symmetry.png", dpi=180)
plt.close(fig)


# -----------------------------------------------------------------------------
# 实验2：深层 ReLU 网络中，不同初始化尺度如何影响 Activation 方差
# -----------------------------------------------------------------------------
section("实验2：初始化尺度与深层网络 Activation 方差")

batch = 2048
width = 256
layers = 30
X0 = torch.randn(batch, width)


def run_deep_relu(init_name, scale_fn):
    """构造多层 Linear+ReLU，不训练，只观察每层 activation std。"""
    x_cur = X0.clone()
    act_stds = []
    for _ in range(layers):
        W = torch.randn(width, width) * scale_fn(width)
        x_cur = torch.relu(x_cur @ W)
        act_stds.append(float(x_cur.std()))
    return np.array(act_stds)

# 过小初始化：固定 std=0.01
small = run_deep_relu("small", lambda fan_in: 0.01)
# 过大初始化：固定 std=0.20
large = run_deep_relu("large", lambda fan_in: 0.20)
# Xavier normal 的近似尺度：sqrt(1/fan_in)
xavier = run_deep_relu("xavier", lambda fan_in: math.sqrt(1.0 / fan_in))
# He/Kaiming normal 对 ReLU 常见尺度：sqrt(2/fan_in)
he = run_deep_relu("he", lambda fan_in: math.sqrt(2.0 / fan_in))

for name, arr in [("small std=0.01", small), ("large std=0.20", large), ("Xavier", xavier), ("He", he)]:
    print(f"{name:>16s}: layer1 std={arr[0]:.6g}, layer10 std={arr[9]:.6g}, layer30 std={arr[-1]:.6g}")

fig, ax = plt.subplots(figsize=(8.5, 5.2))
for name, arr in [("过小 std=0.01", small), ("过大 std=0.20", large), ("Xavier", xavier), ("He / Kaiming", he)]:
    # 避免 log(0)；极端爆炸用有限值显示
    ax.plot(np.arange(1, layers+1), np.maximum(arr, 1e-30), label=name)
ax.set_yscale("log")
ax.set_xlabel("Layer")
ax.set_ylabel("Activation Std (log scale)")
ax.set_title("不同初始化尺度在 30 层 ReLU 网络中的 Activation 尺度")
ax.legend()
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(FIG_DIR / "lesson40_activation_scale_initialization.png", dpi=180)
plt.close(fig)


# -----------------------------------------------------------------------------
# 实验3：链式法则中的梯度消失 / 梯度爆炸
# -----------------------------------------------------------------------------
section("实验3：Chain Rule 中的 Vanishing / Exploding Gradient")

depths = np.arange(1, 31)
for a in [0.5, 0.9, 1.0, 1.1, 1.5]:
    grad = a ** depths
    print(f"局部导数 a={a}: depth=10 -> {grad[9]:.6g}, depth=20 -> {grad[19]:.6g}, depth=30 -> {grad[29]:.6g}")

fig, ax = plt.subplots(figsize=(8.5, 5.2))
for a in [0.5, 0.9, 1.0, 1.1, 1.5]:
    ax.plot(depths, np.abs(a ** depths), label=f"局部导数={a}")
ax.set_yscale("log")
ax.set_xlabel("连续相乘的深度")
ax.set_ylabel("|Gradient multiplier| (log scale)")
ax.set_title("链式法则：小于 1 的局部导数会消失，大于 1 会爆炸")
ax.legend()
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(FIG_DIR / "lesson40_vanishing_exploding_chain.png", dpi=180)
plt.close(fig)


# -----------------------------------------------------------------------------
# 实验4：Tanh 饱和区导致小导数
# -----------------------------------------------------------------------------
section("实验4：Tanh Saturation 与 Gradient")

z = torch.tensor([-5.0, -2.0, 0.0, 2.0, 5.0], requires_grad=True)
a = torch.tanh(z)
a.sum().backward()
print("z:", z.detach().numpy())
print("tanh(z):", a.detach().numpy())
print("d tanh / dz:", z.grad.detach().numpy())

zs = np.linspace(-6, 6, 400)
tanh_vals = np.tanh(zs)
tanh_der = 1 - tanh_vals ** 2
fig, ax = plt.subplots(figsize=(8.5, 5.2))
ax.plot(zs, tanh_vals, label="tanh(z)")
ax.plot(zs, tanh_der, label="tanh'(z)")
ax.set_xlabel("z")
ax.set_ylabel("Value")
ax.set_title("Tanh 饱和区：输出接近 ±1 时导数接近 0")
ax.legend()
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(FIG_DIR / "lesson40_tanh_saturation_derivative.png", dpi=180)
plt.close(fig)


# -----------------------------------------------------------------------------
# 实验5：Gradient Clipping 的作用
# -----------------------------------------------------------------------------
section("实验5：Gradient Clipping")

p1 = nn.Parameter(torch.tensor([100.0, -50.0, 25.0]))
p2 = nn.Parameter(torch.tensor([30.0]))
p1.grad = torch.tensor([120.0, -160.0, 80.0])
p2.grad = torch.tensor([60.0])
params = [p1, p2]

before = math.sqrt(sum(float((p.grad ** 2).sum()) for p in params))
max_norm = 10.0
reported_before = torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm)
after = math.sqrt(sum(float((p.grad ** 2).sum()) for p in params))

print(f"clipping 前 global grad norm = {before:.6f}")
print(f"clip_grad_norm_ 返回的原始 norm = {float(reported_before):.6f}")
print(f"clipping 后 global grad norm = {after:.6f}")
print("clipping 后 p1.grad =", p1.grad.detach().numpy())
print("clipping 后 p2.grad =", p2.grad.detach().numpy())

fig, ax = plt.subplots(figsize=(7.5, 4.8))
ax.bar(["clipping 前", "clipping 后"], [before, after])
ax.axhline(max_norm, linestyle="--", linewidth=1.5, label=f"max_norm={max_norm}")
ax.set_ylabel("Global Gradient Norm")
ax.set_title("Gradient Clipping 限制全局梯度范数")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "lesson40_gradient_clipping.png", dpi=180)
plt.close(fig)


# -----------------------------------------------------------------------------
# 实验6：PyTorch 官方初始化 API 的尺度检查
# -----------------------------------------------------------------------------
section("实验6：PyTorch Xavier / Kaiming 初始化 API")

W_xavier = torch.empty(512, 512)
W_he = torch.empty(512, 512)
nn.init.xavier_normal_(W_xavier)
nn.init.kaiming_normal_(W_he, nonlinearity="relu")
print(f"xavier_normal_ 实际 std = {float(W_xavier.std()):.6f}, 理论量级 ~ {math.sqrt(1/512):.6f}")
print(f"kaiming_normal_ 实际 std = {float(W_he.std()):.6f}, 理论量级 ~ {math.sqrt(2/512):.6f}")

print("\n所有实验已完成。图像保存在:", FIG_DIR)
