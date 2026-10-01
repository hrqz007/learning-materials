# -*- coding: utf-8 -*-
"""
第09讲实验：Activation Function
目标：
1. 用 NumPy 手写 ReLU / Sigmoid / Tanh / GELU / SiLU。
2. 观察相同输入经过不同激活函数后，数值如何变化。
3. 验证激活函数逐元素工作，因此通常不改变 Tensor Shape。
4. 观察加入非线性后，网络不再能简单折叠为一个 affine 变换。
5. 可选：用 PyTorch 验证 NumPy 结果。
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# 0. 输出目录与打印设置
# -----------------------------
OUT_DIR = Path(__file__).resolve().parent
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)
np.set_printoptions(precision=6, suppress=True)


# -----------------------------
# 1. 手写五种激活函数
# -----------------------------
def relu(x):
    """ReLU(x) = max(0, x)，逐元素计算。"""
    return np.maximum(0.0, x)


def sigmoid(x):
    """Sigmoid(x) = 1 / (1 + exp(-x))。"""
    return 1.0 / (1.0 + np.exp(-x))


def tanh(x):
    """Tanh：直接调用 NumPy 的双曲正切实现。"""
    return np.tanh(x)


def gelu(x):
    """
    GELU 的常用 tanh 近似：
    0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
    """
    return 0.5 * x * (
        1.0
        + np.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * np.power(x, 3))
        )
    )


def silu(x):
    """SiLU(x) = x * sigmoid(x)。"""
    return x * sigmoid(x)


# -----------------------------
# 2. 实验一：几个具体数字经过五种激活函数
# -----------------------------
print("=" * 70)
print("实验一：具体数字经过五种激活函数")
print("=" * 70)

x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
print("输入 x       =", x)
print("ReLU(x)      =", relu(x))
print("Sigmoid(x)   =", sigmoid(x))
print("Tanh(x)      =", tanh(x))
print("GELU(x)      =", gelu(x))
print("SiLU(x)      =", silu(x))
print()


# -----------------------------
# 3. 实验二：画激活函数曲线
# -----------------------------
print("=" * 70)
print("实验二：生成激活函数曲线")
print("=" * 70)

xs = np.linspace(-5.0, 5.0, 1000)

plt.figure(figsize=(8, 5))
plt.plot(xs, relu(xs), label="ReLU")
plt.plot(xs, sigmoid(xs), label="Sigmoid")
plt.plot(xs, tanh(xs), label="Tanh")
plt.plot(xs, gelu(xs), label="GELU")
plt.plot(xs, silu(xs), label="SiLU")
plt.axhline(0.0, linewidth=0.8)
plt.axvline(0.0, linewidth=0.8)
plt.xlabel("x")
plt.ylabel("activation(x)")
plt.title("Five Activation Functions")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
fig_all = FIG_DIR / "lesson09_activation_curves.png"
plt.savefig(fig_all, dpi=180)
plt.close()
print("已保存：", fig_all.name)

# 进一步放大 0 附近，便于比较 ReLU/GELU/SiLU。
xs2 = np.linspace(-3.0, 3.0, 1000)
plt.figure(figsize=(8, 5))
plt.plot(xs2, relu(xs2), label="ReLU")
plt.plot(xs2, gelu(xs2), label="GELU")
plt.plot(xs2, silu(xs2), label="SiLU")
plt.axhline(0.0, linewidth=0.8)
plt.axvline(0.0, linewidth=0.8)
plt.xlabel("x")
plt.ylabel("activation(x)")
plt.title("ReLU vs GELU vs SiLU near zero")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
fig_modern = FIG_DIR / "lesson09_relu_gelu_silu.png"
plt.savefig(fig_modern, dpi=180)
plt.close()
print("已保存：", fig_modern.name)
print()


# -----------------------------
# 4. 实验三：Linear 输出经过激活函数，Shape 是否改变？
# -----------------------------
print("=" * 70)
print("实验三：Linear -> Activation，观察数值与 Shape")
print("=" * 70)

# 两个样本，每个样本 3 个输入特征。
X = np.array([
    [1.0, 2.0, -1.0],
    [-2.0, 0.5, 3.0],
])

# 3 个输入特征 -> 4 个输出特征。
W = np.array([
    [0.5, -1.0, 0.3, 0.2],
    [1.0,  0.4, -0.6, 0.8],
    [-0.5, 0.7,  1.2, -1.0],
])
b = np.array([0.1, -0.2, 0.0, 0.5])

Z = X @ W + b
A_relu = relu(Z)
A_gelu = gelu(Z)
A_silu = silu(Z)

print("X.shape       =", X.shape)
print("W.shape       =", W.shape)
print("b.shape       =", b.shape)
print("Z.shape       =", Z.shape)
print("ReLU(Z).shape =", A_relu.shape)
print("GELU(Z).shape =", A_gelu.shape)
print("SiLU(Z).shape =", A_silu.shape)
print("\nLinear 输出 Z =\n", Z)
print("\nReLU(Z) =\n", A_relu)
print("\nGELU(Z) =\n", A_gelu)
print("\nSiLU(Z) =\n", A_silu)
print()


# -----------------------------
# 5. 实验四：加入 ReLU 后，不能再简单折叠成一个 affine
# -----------------------------
print("=" * 70)
print("实验四：加入 ReLU 后，单个 affine 无法精确复现")
print("=" * 70)

# 一维输入，网络 y = 2*ReLU(3x - 1) + 0.5
x_line = np.linspace(-2.0, 2.0, 101)
y_nonlinear = 2.0 * relu(3.0 * x_line - 1.0) + 0.5

# 用最小二乘拟合最佳单一直线 y = ax + c。
A = np.column_stack([x_line, np.ones_like(x_line)])
a_best, c_best = np.linalg.lstsq(A, y_nonlinear, rcond=None)[0]
y_affine = a_best * x_line + c_best
max_error = np.max(np.abs(y_nonlinear - y_affine))

print(f"最佳单一直线：y = {a_best:.6f} * x + {c_best:.6f}")
print(f"最大绝对误差：{max_error:.6f}")
print("若能完全折叠为 affine，最大误差应接近 0；这里明显不为 0。")
print()

plt.figure(figsize=(8, 5))
plt.plot(x_line, y_nonlinear, label="2*ReLU(3x-1)+0.5")
plt.plot(x_line, y_affine, linestyle="--", label="best single affine fit")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Activation breaks affine collapse")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
fig_break = FIG_DIR / "lesson09_activation_breaks_affine.png"
plt.savefig(fig_break, dpi=180)
plt.close()
print("已保存：", fig_break.name)
print()


# -----------------------------
# 6. 实验五：可选 PyTorch 对照
# -----------------------------
print("=" * 70)
print("实验五：PyTorch 对照（若已安装 torch）")
print("=" * 70)

try:
    import torch
    import torch.nn.functional as F

    z_t = torch.tensor(Z, dtype=torch.float64)
    np_relu = A_relu
    np_gelu = gelu(Z)
    np_silu = silu(Z)

    torch_relu = F.relu(z_t).numpy()
    torch_gelu = F.gelu(z_t, approximate="tanh").numpy()
    torch_silu = F.silu(z_t).numpy()

    print("NumPy ReLU 与 PyTorch ReLU 最大差：", np.max(np.abs(np_relu - torch_relu)))
    print("NumPy GELU 与 PyTorch tanh-GELU 最大差：", np.max(np.abs(np_gelu - torch_gelu)))
    print("NumPy SiLU 与 PyTorch SiLU 最大差：", np.max(np.abs(np_silu - torch_silu)))
except Exception as e:
    print("未运行 PyTorch 对照。原因：", repr(e))

print("\n实验完成。")
