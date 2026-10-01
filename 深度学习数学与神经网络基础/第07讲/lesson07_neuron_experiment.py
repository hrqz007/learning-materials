# 第07讲实验：神经元究竟是什么
# 目标：把“一个神经元”拆成可手算、可验证的数值过程：输入 -> 权重 -> 加权贡献 -> 求和 -> bias -> z。

import numpy as np

np.set_printoptions(precision=4, suppress=True)

print("=" * 78)
print("实验1：手算一个神经元的加权和 z = x·w + b")
print("=" * 78)

# 一个样本有 3 个输入特征，因此 x.shape = (3,)
x = np.array([2.0, -1.0, 3.0])

# 一个神经元需要为每个输入特征准备一个权重，因此 w.shape = (3,)
w = np.array([0.5, -2.0, 1.0])

# 一个神经元只有一个标量 bias。
b = 0.2

# 每个输入特征各自产生一个“加权贡献”。
contributions = x * w

# 把所有贡献相加，再加 bias，得到神经元激活函数之前的值 z。
z = contributions.sum() + b

print("x =", x, "shape =", x.shape)
print("w =", w, "shape =", w.shape)
print("b =", b)
print("逐项贡献 x*w =", contributions)
print("贡献之和 =", contributions.sum())
print("z = x·w + b =", z)
print("np.dot(x, w) + b =", np.dot(x, w) + b)

print("\n" + "=" * 78)
print("实验2：只改变一个输入特征，观察 z 怎样变化")
print("=" * 78)

# 将第 1 个输入特征从 2.0 增加到 4.0，其余保持不变。
x_changed = x.copy()
x_changed[0] = 4.0
z_changed = x_changed @ w + b

print("原始 x =", x, "原始 z =", z)
print("改变后 x =", x_changed, "改变后 z =", z_changed)
print("z 的变化量 =", z_changed - z)
print("理论变化量 = Δx1 * w1 =", (x_changed[0] - x[0]) * w[0])
print("结论：某个输入对 z 的影响大小与对应 weight 直接相关。")

print("\n" + "=" * 78)
print("实验3：Weight 的正负号决定输入增加时 z 往哪个方向变化")
print("=" * 78)

# 逐个把三个输入都增加 1，观察 z 的变化。
for i in range(3):
    x_temp = x.copy()
    x_temp[i] += 1.0
    z_temp = x_temp @ w + b
    print(f"特征{i+1} 增加 1：weight={w[i]: .2f}, z变化={z_temp-z: .2f}")

print("说明：正权重会使该输入增加时 z 增大；负权重会使 z 减小；绝对值越大，单位变化影响越大。")

print("\n" + "=" * 78)
print("实验4：Bias 像整体平移项，不依赖某一个输入特征")
print("=" * 78)

for bias_value in [-2.0, 0.0, 0.2, 3.0]:
    z_bias = x @ w + bias_value
    print(f"b={bias_value: .1f} -> z={z_bias: .2f}")

print("结论：改变 bias 会整体移动神经元的 pre-activation z。")

print("\n" + "=" * 78)
print("实验5：多个神经元组成一层，本质就是矩阵乘法")
print("=" * 78)

# 现在不只要 1 个神经元，而是同时要 2 个神经元。
# 每一列对应一个神经元的权重向量，所以 W.shape = (3, 2)。
W = np.array([[0.5, -1.0],
              [-2.0, 0.5],
              [1.0, 2.0]])

# 两个神经元各自有一个 bias。
b_vec = np.array([0.2, -0.3])

# 一次矩阵乘法即可得到两个神经元的 z。
z_layer = x @ W + b_vec

# 为了验证，逐个神经元单独算一次。
z_neuron_1 = x @ W[:, 0] + b_vec[0]
z_neuron_2 = x @ W[:, 1] + b_vec[1]

print("W shape =", W.shape)
print("b_vec shape =", b_vec.shape)
print("整层一次计算 z_layer =", z_layer)
print("逐神经元计算 =", np.array([z_neuron_1, z_neuron_2]))
print("两种方式完全一致：", np.allclose(z_layer, [z_neuron_1, z_neuron_2]))

print("\n" + "=" * 78)
print("实验6：Batch 中的多个样本共享同一个神经元/同一层参数")
print("=" * 78)

X = np.array([[2.0, -1.0, 3.0],
              [0.0,  1.0, 2.0],
              [1.0,  1.0, 1.0]])

Z_batch = X @ W + b_vec

print("X shape =", X.shape)
print("W shape =", W.shape)
print("Z_batch shape =", Z_batch.shape)
print("Z_batch =\n", Z_batch)
print("第一个样本结果与实验5一致：", np.allclose(Z_batch[0], z_layer))

print("\n" + "=" * 78)
print("实验7：权重大小不等于现实世界中的“因果重要性”")
print("=" * 78)

# 构造两个数值尺度完全不同的特征，展示直接比较 weight 可能误导。
x_scale = np.array([1000.0, 1.0])
w_scale = np.array([0.001, 0.8])
contrib_scale = x_scale * w_scale

print("x =", x_scale)
print("w =", w_scale)
print("逐项贡献 x*w =", contrib_scale)
print("虽然 w1=0.001 远小于 w2=0.8，但当前样本中两项贡献分别为", contrib_scale)
print("结论：weight 的解释依赖输入尺度、网络上下文与训练过程，不能简单等同于现实因果重要性。")

print("\n" + "=" * 78)
print("实验8：可选 PyTorch 对照 - nn.Linear 的 pre-activation")
print("=" * 78)

try:
    import torch
    import torch.nn as nn

    layer = nn.Linear(3, 2, bias=True)

    # PyTorch 的 weight 保存为 (out_features, in_features)，因此复制 W.T。
    with torch.no_grad():
        layer.weight.copy_(torch.tensor(W.T, dtype=torch.float32))
        layer.bias.copy_(torch.tensor(b_vec, dtype=torch.float32))

    x_t = torch.tensor(x, dtype=torch.float32)
    z_t = layer(x_t)

    print("PyTorch weight shape =", tuple(layer.weight.shape))
    print("PyTorch bias shape   =", tuple(layer.bias.shape))
    print("PyTorch 输出 z       =", z_t.detach().numpy())
    print("与 NumPy 一致：", np.allclose(z_t.detach().numpy(), z_layer, atol=1e-6))
except Exception as e:
    print("PyTorch 对照未执行：", repr(e))
    print("这不影响本讲 NumPy 主实验。")

print("\n" + "=" * 78)
print("实验结束：本讲必须记住的结论")
print("=" * 78)
print("1. 一个最基本神经元先计算 z = x·w + b。")
print("2. 每个输入都有对应 weight；x_i*w_i 是该输入对当前 z 的直接数值贡献。")
print("3. 多个神经元并排，就是把多个 w 堆成矩阵 W，一次做 x@W+b。")
print("4. Batch 中所有样本共享同一套 W、b。")
print("5. 目前的 z 仍只是线性/仿射结果；下一步还需要 Activation 才能构成真正有非线性表达能力的深层网络。")
