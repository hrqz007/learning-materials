import numpy as np

# ============================================================
# 第12讲实验：不用 PyTorch，使用 NumPy 从头完成一个两层 MLP 的 Forward
# 网络结构：3 -> 4 -> 2
# ============================================================

# 为了让打印结果更容易阅读，保留 3 位小数并关闭科学计数法。
np.set_printoptions(precision=3, suppress=True)


def linear(x, weight, bias):
    """线性/仿射层：输入 x 乘权重 weight，再加 bias。"""
    return x @ weight + bias


def relu(x):
    """ReLU：把负数变成 0，非负数保持不变。"""
    return np.maximum(0.0, x)


def forward(x, W1, b1, W2, b2, verbose=True):
    """两层 MLP 的完整前向传播。"""
    # 第一层 Linear：最后一维从 3 变成 4。
    z1 = linear(x, W1, b1)

    # 激活函数：只改变数值，不改变 Shape。
    a1 = relu(z1)

    # 第二层 Linear：最后一维从 4 变成 2。
    y = linear(a1, W2, b2)

    if verbose:
        print("输入 x：")
        print(x)
        print("x.shape =", x.shape)
        print()

        print("第一层输出 z1 = x @ W1 + b1：")
        print(z1)
        print("z1.shape =", z1.shape)
        print()

        print("ReLU 后 a1 = relu(z1)：")
        print(a1)
        print("a1.shape =", a1.shape)
        print()

        print("最终输出 y = a1 @ W2 + b2：")
        print(y)
        print("y.shape =", y.shape)

    return y, {"z1": z1, "a1": a1}


# ------------------------------------------------------------
# 1. 定义网络参数
# ------------------------------------------------------------
# W1.shape = (3, 4)：3 个输入特征 -> 4 个隐藏神经元。
W1 = np.array([
    [0.5, -1.0,  0.0, 1.0],
    [1.0,  0.5, -0.5, 0.0],
    [-0.5, 1.0,  1.0, 0.5],
], dtype=np.float64)

# b1.shape = (4,)：第一层 4 个隐藏神经元各有 1 个 bias。
b1 = np.array([0.1, -0.2, 0.3, 0.0], dtype=np.float64)

# W2.shape = (4, 2)：4 个隐藏特征 -> 2 个输出。
W2 = np.array([
    [1.0, -0.5],
    [0.5,  1.0],
    [-1.0, 0.5],
    [0.2,  0.8],
], dtype=np.float64)

# b2.shape = (2,)：最终 2 个输出各有 1 个 bias。
b2 = np.array([0.1, -0.2], dtype=np.float64)

print("=" * 72)
print("实验 1：检查参数 Shape 和参数数量")
print("=" * 72)
print("W1.shape =", W1.shape)
print("b1.shape =", b1.shape)
print("W2.shape =", W2.shape)
print("b2.shape =", b2.shape)
parameter_count = W1.size + b1.size + W2.size + b2.size
print("总参数量 =", parameter_count)
print("理论计算 = 3*4 + 4 + 4*2 + 2 =", 3*4 + 4 + 4*2 + 2)
print()

# ------------------------------------------------------------
# 2. 单样本 Forward
# ------------------------------------------------------------
print("=" * 72)
print("实验 2：单个样本完整 Forward")
print("=" * 72)
x = np.array([1.0, 2.0, -1.0], dtype=np.float64)
y, cache = forward(x, W1, b1, W2, b2, verbose=True)
print()

# ------------------------------------------------------------
# 3. 手工核对第一个隐藏神经元
# ------------------------------------------------------------
print("=" * 72)
print("实验 3：手工核对第一个隐藏神经元")
print("=" * 72)
manual_h1 = (
    x[0] * W1[0, 0]
    + x[1] * W1[1, 0]
    + x[2] * W1[2, 0]
    + b1[0]
)
print("手工：1*0.5 + 2*1.0 + (-1)*(-0.5) + 0.1 =", manual_h1)
print("NumPy z1[0] =", cache["z1"][0])
print("是否一致 =", np.allclose(manual_h1, cache["z1"][0]))
print()

# ------------------------------------------------------------
# 4. Batch Forward
# ------------------------------------------------------------
print("=" * 72)
print("实验 4：多个样本组成 Batch，一次完成 Forward")
print("=" * 72)
X = np.array([
    [1.0,  2.0, -1.0],
    [0.5, -1.0,  2.0],
    [-2.0, 1.0,  0.5],
], dtype=np.float64)
Y_batch, cache_batch = forward(X, W1, b1, W2, b2, verbose=False)
print("X.shape =", X.shape)
print("z1.shape =", cache_batch["z1"].shape)
print("a1.shape =", cache_batch["a1"].shape)
print("Y_batch.shape =", Y_batch.shape)
print("Y_batch =")
print(Y_batch)
print()

# ------------------------------------------------------------
# 5. Batch 计算是否等于逐样本计算
# ------------------------------------------------------------
print("=" * 72)
print("实验 5：Batch 结果与逐样本结果是否相同")
print("=" * 72)
Y_loop = []
for sample in X:
    sample_y, _ = forward(sample, W1, b1, W2, b2, verbose=False)
    Y_loop.append(sample_y)
Y_loop = np.stack(Y_loop, axis=0)
print("逐样本计算结果 =")
print(Y_loop)
print("最大绝对误差 =", np.max(np.abs(Y_batch - Y_loop)))
print("是否一致 =", np.allclose(Y_batch, Y_loop))
print()

# ------------------------------------------------------------
# 6. 参数扰动：只改一个权重，看中间量和最终输出如何变化
# ------------------------------------------------------------
print("=" * 72)
print("实验 6：只改变一个 Weight，观察影响怎样沿 Forward 路径传播")
print("=" * 72)
W1_changed = W1.copy()
W1_changed[0, 0] += 1.0  # 只改变输入特征0 -> 隐藏神经元0 的权重

y_old, cache_old = forward(x, W1, b1, W2, b2, verbose=False)
y_new, cache_new = forward(x, W1_changed, b1, W2, b2, verbose=False)

print("原 z1 =", cache_old["z1"])
print("新 z1 =", cache_new["z1"])
print("z1 变化 =", cache_new["z1"] - cache_old["z1"])
print("原 y =", y_old)
print("新 y =", y_new)
print("y 变化 =", y_new - y_old)
print()

# ------------------------------------------------------------
# 7. 再强调：Forward != Training
# ------------------------------------------------------------
print("=" * 72)
print("实验 7：概念检查 - 现在这个网络学习了吗？")
print("=" * 72)
print("没有。我们只是给定参数后完成了 Forward。")
print("要让网络学习，还需要：Target -> Loss -> Gradient -> Backprop -> Optimizer -> 更新参数。")
