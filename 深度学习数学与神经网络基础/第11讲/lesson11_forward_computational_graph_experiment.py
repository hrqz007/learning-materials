# 第11讲实验：Forward Propagation 与 Computational Graph
# 目标：不把 model(x) 当成黑盒，而是显式观察每一个中间量、Shape 和依赖关系。

import numpy as np

np.set_printoptions(precision=4, suppress=True)

print("=" * 70)
print("实验 1：最小标量 Forward")
print("=" * 70)

x = 2.0                       # 输入标量
z = 2.0 * x - 1.0             # 第一个中间节点
# ReLU：负数变 0，正数保持不变
a = max(0.0, z)               # 第二个中间节点
y = 3.0 * a + 0.5             # 最终输出

print(f"x = {x}")
print(f"z = 2*x - 1 = {z}")
print(f"a = ReLU(z) = {a}")
print(f"y = 3*a + 0.5 = {y}")
print("依赖链：x -> z -> a -> y")

print("\n" + "=" * 70)
print("实验 2：两层 MLP 的完整 Forward（沿用第10讲参数）")
print("=" * 70)

# 输入：3 个特征
x = np.array([1.0, 2.0, -1.0])

# 第一层参数：3 -> 4
W1 = np.array([
    [1.0,  0.0, -1.0,  0.5],
    [0.5,  1.0,  0.0, -1.0],
    [1.0, -0.5,  1.0,  1.0],
])
b1 = np.array([0.0, 0.5, 1.0, -0.5])

# 第二层参数：4 -> 2
W2 = np.array([
    [ 1.0, -1.0],
    [ 0.5,  0.5],
    [-1.0,  1.0],
    [ 2.0,  0.0],
])
b2 = np.array([0.2, -0.3])

# Forward：把每一步中间量都显式保存下来
z1 = x @ W1 + b1              # Linear 1
a1 = np.maximum(0.0, z1)      # ReLU
y = a1 @ W2 + b2              # Linear 2

print("x  =", x, "shape =", x.shape)
print("z1 =", z1, "shape =", z1.shape)
print("a1 =", a1, "shape =", a1.shape)
print("y  =", y, "shape =", y.shape)
print("Forward 链：x -> z1 -> a1 -> y")

print("\n" + "=" * 70)
print("实验 3：输入发生有限扰动，观察变化怎样沿 Forward 路径传播")
print("=" * 70)

x_new = x.copy()               # 复制原输入，避免直接修改原数组
x_new[0] += 0.1                # 只把第 0 个特征从 1.0 改成 1.1

z1_new = x_new @ W1 + b1       # 新的第一层输出
a1_new = np.maximum(0.0, z1_new)
y_new = a1_new @ W2 + b2

print("原 x      =", x)
print("新 x      =", x_new)
print("Delta z1  =", z1_new - z1)
print("Delta a1  =", a1_new - a1)
print("Delta y   =", y_new - y)
print("注意：这是两次 Forward 的差，不是 Gradient。")

print("\n" + "=" * 70)
print("实验 4：ReLU 怎样让某条 Forward 路径暂时不传递数值")
print("=" * 70)

for z_test in [-1.0, -0.5, 0.5, 2.0]:
    a_test = max(0.0, z_test)  # ReLU
    y_test = 2.0 * a_test      # 下游节点
    print(f"z={z_test:>4.1f} -> ReLU(z)={a_test:>3.1f} -> y={y_test:>3.1f}")

print("\n" + "=" * 70)
print("实验 5：Batch Forward - 同一个计算图处理多条样本")
print("=" * 70)

X = np.array([
    [1.0,  2.0, -1.0],
    [0.0,  1.0,  2.0],
    [2.0, -1.0,  0.5],
])
Z1 = X @ W1 + b1               # (3,3) @ (3,4) -> (3,4)
A1 = np.maximum(0.0, Z1)       # Shape 不变
Y = A1 @ W2 + b2               # (3,4) @ (4,2) -> (3,2)

print("X.shape  =", X.shape)
print("Z1.shape =", Z1.shape)
print("A1.shape =", A1.shape)
print("Y.shape  =", Y.shape)
print("Y =\n", Y)

print("\n" + "=" * 70)
print("实验 6：如果安装了 PyTorch，验证 nn.Sequential 与手工 Forward 一致")
print("=" * 70)

try:
    import torch
    import torch.nn as nn

    model = nn.Sequential(
        nn.Linear(3, 4),
        nn.ReLU(),
        nn.Linear(4, 2),
    )

    # 把 PyTorch 参数设置成与 NumPy 完全相同。
    # PyTorch Linear 的 weight 保存 Shape 是 (out_features, in_features)，
    # 因此需要把 NumPy 的 W 转置后复制进去。
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor(W1.T, dtype=torch.float32))
        model[0].bias.copy_(torch.tensor(b1, dtype=torch.float32))
        model[2].weight.copy_(torch.tensor(W2.T, dtype=torch.float32))
        model[2].bias.copy_(torch.tensor(b2, dtype=torch.float32))

    x_t = torch.tensor(x, dtype=torch.float32)

    # 不直接只写 model(x)，而是把每个中间量显式打印出来。
    z1_t = model[0](x_t)
    a1_t = model[1](z1_t)
    y_t = model[2](a1_t)

    print("PyTorch z1 =", z1_t.detach().numpy())
    print("PyTorch a1 =", a1_t.detach().numpy())
    print("PyTorch y  =", y_t.detach().numpy())
    print("NumPy y    =", y)
    print("max|PyTorch-NumPy| =", float(np.max(np.abs(y_t.detach().numpy() - y))))
except ImportError:
    print("当前环境没有安装 PyTorch，跳过实验 6；前 5 个实验不受影响。")

print("\n实验结束。")
