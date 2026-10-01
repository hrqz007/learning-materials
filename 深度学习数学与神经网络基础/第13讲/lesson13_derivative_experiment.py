# -*- coding: utf-8 -*-
"""
第13讲实验：函数、斜率与导数
目标：不用符号求导库，先用“很小的变化量”观察导数的数值含义。
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams["axes.unicode_minus"] = False

# 当前脚本所在目录，用于稳定保存实验图。
BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)


def numerical_derivative(f, x, h=1e-5):
    """用中心差分近似 f 在 x 处的导数。"""
    return (f(x + h) - f(x - h)) / (2 * h)


def section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# -----------------------------------------------------------------------------
# 实验1：直线 y = 3x + 2 的斜率为什么始终是 3
# -----------------------------------------------------------------------------
section("实验1：直线 y = 3x + 2 的斜率")

def f_linear(x):
    return 3 * x + 2

for x in [-2.0, 0.0, 4.0]:
    d = numerical_derivative(f_linear, x)
    print(f"x={x:>4.1f}，数值导数≈{d:.6f}")
print("理论斜率 = 3；不同位置得到的导数都接近 3。")


# -----------------------------------------------------------------------------
# 实验2：曲线 y = x^2 的斜率会随位置变化
# -----------------------------------------------------------------------------
section("实验2：曲线 y = x^2 的局部斜率")

def f_square(x):
    return x ** 2

for x in [-2.0, -1.0, 0.0, 1.0, 3.0]:
    d = numerical_derivative(f_square, x)
    theoretical = 2 * x
    print(
        f"x={x:>4.1f}，数值导数≈{d:>9.6f}，"
        f"理论导数 2x={theoretical:>6.2f}"
    )


# -----------------------------------------------------------------------------
# 实验3：h 取得越来越小时，割线斜率如何逼近切线斜率
# -----------------------------------------------------------------------------
section("实验3：从割线斜率逼近导数")
x0 = 3.0
true_derivative = 2 * x0
for h in [1.0, 0.5, 0.1, 0.01, 0.001, 1e-5]:
    # 这里故意使用前向差分，便于直观看“从两点斜率逼近局部斜率”。
    slope = (f_square(x0 + h) - f_square(x0)) / h
    print(
        f"h={h:<8g}  割线斜率≈{slope:>10.6f}  "
        f"与真实导数6的误差={abs(slope - true_derivative):.6f}"
    )


# -----------------------------------------------------------------------------
# 实验4：一个神经网络 Weight 改一点，输出会怎样变化
# -----------------------------------------------------------------------------
section("实验4：Weight 对输出的局部敏感度")
x = 2.5
b = 0.4

def neuron_output(w):
    # 一个最简单的线性神经元：y = w*x + b
    return w * x + b

w0 = 1.2
print(f"输入 x={x}，Bias b={b}，当前 Weight w={w0}")
print(f"当前输出 y={neuron_output(w0):.6f}")
print(f"数值估计 dy/dw≈{numerical_derivative(neuron_output, w0):.6f}")
print(f"理论上 dy/dw=x={x:.6f}")
print("含义：当 w 增加约 0.01 时，输出大约增加 x*0.01 = 0.025。")

for delta_w in [0.1, 0.01, 0.001]:
    observed = neuron_output(w0 + delta_w) - neuron_output(w0)
    predicted = x * delta_w
    print(
        f"Δw={delta_w:<6g}  实际 Δy={observed:.6f}  "
        f"导数预测 Δy≈{predicted:.6f}"
    )


# -----------------------------------------------------------------------------
# 实验5：ReLU 在负区间、正区间和 0 附近的导数直觉
# -----------------------------------------------------------------------------
section("实验5：ReLU 的局部斜率")

def relu(x):
    return np.maximum(0.0, x)

for x in [-2.0, -0.5, 0.5, 2.0]:
    d = numerical_derivative(relu, x)
    print(f"x={x:>4.1f}，ReLU(x)={relu(x):.2f}，数值导数≈{d:.6f}")

# 在 x=0 处，ReLU 左侧斜率为0，右侧斜率为1，严格说导数不存在。
h = 1e-5
left_slope = (relu(0.0) - relu(-h)) / h
right_slope = (relu(h) - relu(0.0)) / h
center_slope = numerical_derivative(relu, 0.0, h=h)
print(f"x=0 左侧斜率≈{left_slope:.6f}")
print(f"x=0 右侧斜率≈{right_slope:.6f}")
print(f"中心差分会给出≈{center_slope:.6f}，但这不代表严格导数存在。")


# -----------------------------------------------------------------------------
# 图1：x^2 曲线、x=3 处切线，以及一条割线
# -----------------------------------------------------------------------------
xs = np.linspace(-1, 5, 400)
ys = f_square(xs)
x0 = 3.0
y0 = f_square(x0)
tangent = y0 + true_derivative * (xs - x0)
h_plot = 1.0
secant_slope = (f_square(x0 + h_plot) - f_square(x0)) / h_plot
secant = y0 + secant_slope * (xs - x0)

plt.figure(figsize=(8, 5.5))
plt.plot(xs, ys, label="y = x^2")
plt.plot(xs, tangent, label="Tangent at x=3, slope=6")
plt.plot(xs, secant, linestyle="--", label="Secant with h=1, slope=7")
plt.scatter([x0, x0 + h_plot], [f_square(x0), f_square(x0 + h_plot)])
plt.xlabel("x")
plt.ylabel("y")
plt.title("From Secant Slope to Local Derivative")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
fig1 = FIG_DIR / "lesson13_secant_to_tangent.png"
plt.savefig(fig1, dpi=180)
plt.close()


# -----------------------------------------------------------------------------
# 图2：ReLU 曲线，帮助理解分段斜率
# -----------------------------------------------------------------------------
xs = np.linspace(-3, 3, 400)
plt.figure(figsize=(8, 5.5))
plt.plot(xs, relu(xs), label="ReLU(x)=max(0,x)")
plt.axvline(0, linewidth=1)
plt.axhline(0, linewidth=1)
plt.xlabel("x")
plt.ylabel("ReLU(x)")
plt.title("ReLU: slope 0 for x<0, slope 1 for x>0")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
fig2 = FIG_DIR / "lesson13_relu_slope.png"
plt.savefig(fig2, dpi=180)
plt.close()

print("\n实验图已保存：")
print(fig1)
print(fig2)
