# 第08讲实验：为什么仅有 Linear 不够
# 目标：通过可手算例子、NumPy、PyTorch 和图形验证：多个 Linear/Affine 层之间若没有非线性操作，
# 在“函数表达能力”上仍等价于一个 Linear/Affine 变换；加入非线性后，这种折叠不再成立。

import os
import numpy as np

np.set_printoptions(precision=4, suppress=True)

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

print("=" * 86)
print("实验1：标量案例 - 两层 affine 可以合并成一层")
print("=" * 86)

# 第一层：h = 2x + 1
# 第二层：y = 3h - 4
# 代入后：y = 3(2x+1)-4 = 6x-1
xs = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
h = 2.0 * xs + 1.0
y_two = 3.0 * h - 4.0
y_one = 6.0 * xs - 1.0

print("x      =", xs)
print("two-layer y =", y_two)
print("collapsed y =", y_one)
print("完全一致：", np.allclose(y_two, y_one))

print("\n" + "=" * 86)
print("实验2：矩阵案例 - 两层 X@W+b 可以精确折叠")
print("=" * 86)

X = np.array([[1.0, 2.0],
              [-1.0, 0.5],
              [2.0, -1.0]])              # shape = (3, 2)
W1 = np.array([[1.0, 2.0, -1.0],
               [0.5, -1.0, 3.0]])        # shape = (2, 3)
b1 = np.array([0.2, -0.3, 0.5])          # shape = (3,)
W2 = np.array([[2.0, -1.0],
               [0.0,  1.5],
               [1.0,  0.5]])             # shape = (3, 2)
b2 = np.array([-0.4, 0.7])               # shape = (2,)

H = X @ W1 + b1
Y_two = H @ W2 + b2

# 对 row-vector 记法：
# (XW1+b1)W2+b2 = X(W1W2) + (b1W2+b2)
W_eq = W1 @ W2
b_eq = b1 @ W2 + b2
Y_one = X @ W_eq + b_eq

print("X shape   =", X.shape)
print("W1 shape  =", W1.shape, "b1 shape =", b1.shape)
print("W2 shape  =", W2.shape, "b2 shape =", b2.shape)
print("H shape   =", H.shape)
print("Y shape   =", Y_two.shape)
print("W_eq shape=", W_eq.shape, "b_eq shape =", b_eq.shape)
print("W_eq =\n", W_eq)
print("b_eq =", b_eq)
print("两层输出 =\n", Y_two)
print("折叠输出 =\n", Y_one)
print("完全一致：", np.allclose(Y_two, Y_one))

print("\n" + "=" * 86)
print("实验3：五层 affine 仍然可以逐层折叠")
print("=" * 86)

rng = np.random.default_rng(42)
dims = [3, 5, 4, 6, 2, 3]
Ws = [rng.normal(size=(dims[i], dims[i+1])) for i in range(len(dims)-1)]
bs = [rng.normal(size=(dims[i+1],)) for i in range(len(dims)-1)]
X5 = rng.normal(size=(7, dims[0]))

Y_deep = X5.copy()
for W, b in zip(Ws, bs):
    Y_deep = Y_deep @ W + b

W_fold = Ws[0].copy()
b_fold = bs[0].copy()
for i in range(1, len(Ws)):
    b_fold = b_fold @ Ws[i] + bs[i]
    W_fold = W_fold @ Ws[i]

Y_fold = X5 @ W_fold + b_fold
print("输入 shape =", X5.shape)
print("五层最终输出 shape =", Y_deep.shape)
print("折叠后 W shape =", W_fold.shape, "b shape =", b_fold.shape)
print("最大绝对误差 =", np.max(np.abs(Y_deep - Y_fold)))
print("完全一致：", np.allclose(Y_deep, Y_fold, atol=1e-10))

print("\n" + "=" * 86)
print("实验4：加入 ReLU 后，不能再用一个 affine 在全局精确替代")
print("=" * 86)

# 这里只把 ReLU 当作下一讲的预告：ReLU(z)=max(0,z)。
# 目标函数：y = 3*ReLU(2x+1)-4
x_curve = np.linspace(-3.0, 3.0, 301)
relu = lambda z: np.maximum(0.0, z)
y_relu = 3.0 * relu(2.0 * x_curve + 1.0) - 4.0

# 找一条“最佳最小二乘直线” y = ax+b 去近似它。
A = np.column_stack([x_curve, np.ones_like(x_curve)])
a_fit, b_fit = np.linalg.lstsq(A, y_relu, rcond=None)[0]
y_fit = a_fit * x_curve + b_fit
max_err = np.max(np.abs(y_relu - y_fit))
mse = np.mean((y_relu - y_fit) ** 2)

print(f"最佳拟合直线：y = {a_fit:.4f} x + {b_fit:.4f}")
print(f"最大绝对误差 = {max_err:.4f}")
print(f"MSE = {mse:.4f}")
print("结论：有 ReLU 后出现“折点”，一条全局直线无法精确重现。")

print("\n" + "=" * 86)
print("实验5：x^2 的三个点证明单一 affine 无法表达简单非线性")
print("=" * 86)

# 若想同时满足 (-1,1), (0,0), (1,1)：
# 从 x=0 得 b=0；从 x=1 得 a=1；此时 x=-1 会预测 -1，而真实是 1。
pts_x = np.array([-1.0, 0.0, 1.0])
pts_y = pts_x ** 2
print("目标点 x =", pts_x)
print("目标点 y=x^2 =", pts_y)
print("若通过 (0,0) 和 (1,1)，直线只能是 y=x")
print("该直线在 x=-1 预测 -1，而目标为 1 -> 无法同时满足三个点。")

print("\n" + "=" * 86)
print("实验6：可选 PyTorch 对照 - 两个 nn.Linear 无激活时可折叠")
print("=" * 86)

try:
    import torch
    import torch.nn as nn

    torch.manual_seed(7)
    l1 = nn.Linear(2, 3, bias=True)
    l2 = nn.Linear(3, 2, bias=True)

    x_t = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        y_t = l2(l1(x_t))

        # PyTorch 存储：weight 为 (out, in)。
        # y = x @ W1.T + b1，再接 W2.T。
        # 折叠后 weight_eq = W2 @ W1，bias_eq = b1 @ W2.T + b2。
        W_eq_t = l2.weight @ l1.weight
        b_eq_t = l1.bias @ l2.weight.T + l2.bias
        y_eq_t = x_t @ W_eq_t.T + b_eq_t

    print("l1.weight shape =", tuple(l1.weight.shape))
    print("l2.weight shape =", tuple(l2.weight.shape))
    print("折叠 weight shape =", tuple(W_eq_t.shape))
    print("PyTorch 两层与折叠结果一致：", bool(torch.allclose(y_t, y_eq_t, atol=1e-6)))
except Exception as e:
    print("PyTorch 对照未执行：", repr(e))
    print("这不影响 NumPy 主实验。")

print("\n" + "=" * 86)
print("实验7：生成两张结果图")
print("=" * 86)

try:
    import matplotlib.pyplot as plt

    # 图1：两层 affine 与折叠后的单层完全重合。
    x_plot = np.linspace(-3.0, 3.0, 200)
    y_two_plot = 3.0 * (2.0 * x_plot + 1.0) - 4.0
    y_one_plot = 6.0 * x_plot - 1.0

    plt.figure(figsize=(7, 4.5))
    plt.plot(x_plot, y_two_plot, label="Two affine layers")
    plt.plot(x_plot, y_one_plot, linestyle="--", label="Collapsed affine")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Two affine layers exactly collapse into one affine map")
    plt.legend()
    plt.tight_layout()
    fig1 = os.path.join(FIG_DIR, "lesson08_affine_collapse.png")
    plt.savefig(fig1, dpi=180)
    plt.close()

    # 图2：带 ReLU 的函数出现折点，最佳直线不能精确拟合。
    plt.figure(figsize=(7, 4.5))
    plt.plot(x_curve, y_relu, label="3*ReLU(2x+1)-4")
    plt.plot(x_curve, y_fit, linestyle="--", label="Best affine fit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("A nonlinear activation breaks affine collapse")
    plt.legend()
    plt.tight_layout()
    fig2 = os.path.join(FIG_DIR, "lesson08_relu_breaks_collapse.png")
    plt.savefig(fig2, dpi=180)
    plt.close()

    print("已生成：", fig1)
    print("已生成：", fig2)
except Exception as e:
    print("绘图未执行：", repr(e))

print("\n" + "=" * 86)
print("实验结束：本讲必须记住的结论")
print("=" * 86)
print("1. 多个 affine/Linear 层之间若没有非线性操作，函数上仍等价于一个 affine 变换。")
print("2. Bias 不会改变这一结论；它只使严格的 linear map 变成 affine map。")
print("3. 深度不自动等于更强的函数表达能力；非线性是关键之一。")
print("4. 加入 ReLU 这类非线性后，简单的层折叠通常不再成立。")
print("5. 深层线性网络即使函数类没变，参数化和训练动力学仍可能不同；不要把“可折叠”误解成“训练完全一样”。")
