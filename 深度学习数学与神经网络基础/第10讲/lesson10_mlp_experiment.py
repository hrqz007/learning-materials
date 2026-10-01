# -*- coding: utf-8 -*-
"""
第10讲实验：MLP / Feed-Forward Neural Network
目标：把 Linear + Activation + Linear 真正组装起来，并验证 Shape、参数量和数值。

运行：
    python lesson10_mlp_experiment.py

依赖：
    numpy
    matplotlib
    torch（可选；若未安装则自动跳过 PyTorch 对照实验）
"""

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)


def relu(x):
    """ReLU：逐元素把负数变成 0。"""
    return np.maximum(x, 0.0)


def print_title(text):
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


# -----------------------------------------------------------------------------
# 实验 1：单样本手工 MLP
# -----------------------------------------------------------------------------
print_title("实验 1：单样本 MLP = Linear -> ReLU -> Linear")

# 一个样本，3 个输入特征，Shape = (3,)
x = np.array([1.0, 2.0, -1.0])

# 第一层：3 -> 4
# NumPy 记法中 W1.shape = (输入维, 输出维) = (3, 4)
W1 = np.array([
    [1.0,  0.0, -1.0,  0.5],
    [0.5,  1.0,  0.0, -1.0],
    [1.0, -0.5,  1.0,  1.0],
])
b1 = np.array([0.0, 0.5, 1.0, -0.5])

# 第二层：4 -> 2
W2 = np.array([
    [ 1.0, -1.0],
    [ 0.5,  0.5],
    [-1.0,  1.0],
    [ 2.0,  0.0],
])
b2 = np.array([0.2, -0.3])

# Forward Pass
z1 = x @ W1 + b1          # 第一层 Linear 输出（激活前）
a1 = relu(z1)             # 隐藏层激活
y = a1 @ W2 + b2          # 第二层 Linear 输出

print("x.shape  =", x.shape, "x =", x)
print("W1.shape =", W1.shape)
print("b1.shape =", b1.shape)
print("z1.shape =", z1.shape, "z1 =", z1)
print("a1.shape =", a1.shape, "a1 =", a1)
print("W2.shape =", W2.shape)
print("b2.shape =", b2.shape)
print("y.shape  =", y.shape, "y =", y)

# 预期：z1 = [1, 3, -1, -3]，ReLU 后 a1 = [1, 3, 0, 0]，最终 y = [2.7, 0.2]


# -----------------------------------------------------------------------------
# 实验 2：逐个隐藏神经元解释第一层到底在算什么
# -----------------------------------------------------------------------------
print_title("实验 2：第一层的 4 个隐藏神经元逐个计算")
for j in range(W1.shape[1]):
    contribution = x * W1[:, j]
    neuron_z = contribution.sum() + b1[j]
    print(f"隐藏神经元 {j}: x*w = {contribution}, 加和+b = {neuron_z:.4f}, ReLU后 = {max(neuron_z, 0.0):.4f}")


# -----------------------------------------------------------------------------
# 实验 3：参数量计算
# -----------------------------------------------------------------------------
print_title("实验 3：MLP 参数量")
layer1_params = W1.size + b1.size
layer2_params = W2.size + b2.size
total_params = layer1_params + layer2_params
print("第一层参数 = 3*4 + 4 =", layer1_params)
print("第二层参数 = 4*2 + 2 =", layer2_params)
print("总参数量   =", total_params)


# -----------------------------------------------------------------------------
# 实验 4：Batch 输入，验证 Shape 沿网络如何变化
# -----------------------------------------------------------------------------
print_title("实验 4：Batch 输入 Shape 追踪")
X = np.array([
    [ 1.0,  2.0, -1.0],
    [ 0.0,  1.0,  2.0],
    [-1.0,  0.5,  1.0],
    [ 2.0, -1.0,  0.0],
])
Z1 = X @ W1 + b1
A1 = relu(Z1)
Y = A1 @ W2 + b2

print("X.shape  =", X.shape)
print("Z1.shape =", Z1.shape)
print("A1.shape =", A1.shape)
print("Y.shape  =", Y.shape)
print("Y =\n", Y)


# -----------------------------------------------------------------------------
# 实验 5：LLM 风格三维输入 (B, T, D)
# -----------------------------------------------------------------------------
print_title("实验 5：LLM 风格三维 Tensor 也可逐 Token 通过同一个 MLP")
X3 = np.array([
    [
        [ 1.0,  2.0, -1.0],
        [ 0.0,  1.0,  2.0],
        [-1.0,  0.5,  1.0],
    ],
    [
        [ 2.0, -1.0,  0.0],
        [ 1.0,  0.0,  1.0],
        [ 0.5,  0.5,  0.5],
    ],
])
Z13 = X3 @ W1 + b1
A13 = relu(Z13)
Y3 = A13 @ W2 + b2
print("X3.shape  =", X3.shape, "   # (B=2, T=3, D_in=3)")
print("Z13.shape =", Z13.shape, "  # hidden=4")
print("Y3.shape  =", Y3.shape, "   # D_out=2")
print("第0个batch第0个token输出 =", Y3[0, 0])
print("是否等于实验1单样本输出？", np.allclose(Y3[0, 0], y))


# -----------------------------------------------------------------------------
# 实验 6：去掉 ReLU 会发生什么？
# -----------------------------------------------------------------------------
print_title("实验 6：有 ReLU 与无 ReLU 的输出对照")
y_with_relu = y
z1_no_act = x @ W1 + b1
y_without_relu = z1_no_act @ W2 + b2
print("有 ReLU：", y_with_relu)
print("无 ReLU：", y_without_relu)
print("差值    ：", y_with_relu - y_without_relu)
print("解释：ReLU 把 z1 中的负数位置清零，第二层收到的信息发生了改变。")


# -----------------------------------------------------------------------------
# 实验 7：生成隐藏层激活对照图
# -----------------------------------------------------------------------------
print_title("实验 7：保存隐藏层激活图")
idx = np.arange(len(z1))
width = 0.36
plt.figure(figsize=(8, 4.8))
plt.bar(idx - width / 2, z1, width, label="z1: before ReLU")
plt.bar(idx + width / 2, a1, width, label="a1: after ReLU")
plt.axhline(0.0, linewidth=1)
plt.xticks(idx, [f"h{i}" for i in idx])
plt.xlabel("Hidden neuron")
plt.ylabel("Value")
plt.title("Lesson 10: Hidden values before/after ReLU")
plt.legend()
plt.tight_layout()
fig_path = "/mnt/data/deep_learning_lesson10/final/figures/lesson10_hidden_activation.png"
plt.savefig(fig_path, dpi=180)
plt.close()
print("图已保存：", fig_path)


# -----------------------------------------------------------------------------
# 实验 8：PyTorch 对照（如果已安装）
# -----------------------------------------------------------------------------
print_title("实验 8：PyTorch nn.Sequential 对照")
try:
    import torch
    import torch.nn as nn

    torch.manual_seed(0)

    # PyTorch Linear 的 weight 保存为 (out_features, in_features)，因此复制时需要转置。
    model = nn.Sequential(
        nn.Linear(3, 4),
        nn.ReLU(),
        nn.Linear(4, 2),
    )

    with torch.no_grad():
        model[0].weight.copy_(torch.tensor(W1.T, dtype=torch.float32))
        model[0].bias.copy_(torch.tensor(b1, dtype=torch.float32))
        model[2].weight.copy_(torch.tensor(W2.T, dtype=torch.float32))
        model[2].bias.copy_(torch.tensor(b2, dtype=torch.float32))

    tx = torch.tensor(x, dtype=torch.float32)
    ty = model(tx).detach().numpy()

    print("NumPy 输出   =", y)
    print("PyTorch 输出 =", ty)
    print("最大绝对误差 =", np.max(np.abs(y - ty)))
    print("model =")
    print(model)

    pytorch_params = sum(p.numel() for p in model.parameters())
    print("PyTorch 参数量 =", pytorch_params)
    print("是否等于手算 26？", pytorch_params == 26)
except Exception as exc:
    print("当前环境无法运行 PyTorch 对照，已跳过。原因：", repr(exc))


print_title("实验完成")
print("如果你真正看懂了这些输出，就已经知道一个最小 MLP 在 Forward Pass 时到底做了什么。")
