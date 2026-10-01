# 第05讲实验：Linear Transformation、Weight 与 Bias
# 目标：用最小数值例子验证 y = x @ W + b，并与 PyTorch nn.Linear 对照。

import numpy as np

np.set_printoptions(precision=4, suppress=True)

print("=" * 70)
print("实验1：单个样本，手算 y = x @ W + b")
print("=" * 70)

# 一个样本，3个输入特征
x = np.array([2.0, -1.0, 3.0])                 # shape: (3,)
# 3个输入特征 -> 2个输出特征，因此 W shape = (3, 2)
W = np.array([[0.5, -1.0],
              [2.0,  0.0],
              [-0.5, 1.5]])                  # shape: (3, 2)
# 每个输出特征各有一个 bias
b = np.array([0.2, -0.3])                    # shape: (2,)

z_no_bias = x @ W

y = z_no_bias + b

print("x =", x, "shape =", x.shape)
print("W =\n", W, "\nshape =", W.shape)
print("b =", b, "shape =", b.shape)
print("x @ W =", z_no_bias)
print("y = x @ W + b =", y, "shape =", y.shape)

# 手工计算两个输出
manual_y0 = 2.0*0.5 + (-1.0)*2.0 + 3.0*(-0.5) + 0.2
manual_y1 = 2.0*(-1.0) + (-1.0)*0.0 + 3.0*1.5 - 0.3
print("手算 y[0] =", manual_y0)
print("手算 y[1] =", manual_y1)
print("手算与 NumPy 一致：", np.allclose(y, [manual_y0, manual_y1]))

print("\n" + "=" * 70)
print("实验2：Batch 一次通过同一组 W 和 b")
print("=" * 70)

X = np.array([[ 2.0, -1.0,  3.0],
              [ 0.0,  1.0,  2.0],
              [-1.0,  2.0,  0.5]])          # shape: (3, 3)
Y_no_bias = X @ W                             # shape: (3, 2)
Y = Y_no_bias + b                             # b 通过 broadcasting 加到每一行

print("X shape =", X.shape)
print("W shape =", W.shape)
print("X @ W shape =", Y_no_bias.shape)
print("b shape =", b.shape)
print("Y shape =", Y.shape)
print("X @ W =\n", Y_no_bias)
print("Y = X @ W + b =\n", Y)
print("第一行是否等于实验1输出：", np.allclose(Y[0], y))

print("\n" + "=" * 70)
print("实验3：只改变 Bias，观察发生什么")
print("=" * 70)

b2 = np.array([10.2, -5.3])
Y_b2 = X @ W + b2
print("原 b =", b)
print("新 b2 =", b2)
print("Y_b2 - Y =\n", Y_b2 - Y)
print("结论：同一个输出维度，对所有样本都平移相同的 bias 差值。")

print("\n" + "=" * 70)
print("实验4：只改变一列 Weight，观察哪个输出受影响")
print("=" * 70)

W2 = W.copy()
W2[:, 0] = W2[:, 0] * 2.0   # 只放大第0列，也就是只改变第0个输出通道使用的权重
Y_W2 = X @ W2 + b
print("W2 =\n", W2)
print("原 Y =\n", Y)
print("新 Y_W2 =\n", Y_W2)
print("差值 Y_W2 - Y =\n", Y_W2 - Y)
print("结论：只改 W 的第0列，主要直接改变第0个输出；第1个输出保持不变。")

print("\n" + "=" * 70)
print("实验5：参数数量")
print("=" * 70)

in_features = 3
out_features = 2
weight_params = in_features * out_features
bias_params = out_features
print("Weight 参数数 =", weight_params)
print("Bias 参数数 =", bias_params)
print("总参数数 =", weight_params + bias_params)

print("\n" + "=" * 70)
print("实验6：与 PyTorch nn.Linear 对照（若已安装 PyTorch）")
print("=" * 70)

try:
    import torch
    import torch.nn as nn

    # PyTorch 的 nn.Linear(3,2) 内部 weight shape 是 (2,3)，
    # 前向计算等价于 x @ weight.T + bias。
    layer = nn.Linear(3, 2, bias=True)

    with torch.no_grad():
        layer.weight.copy_(torch.tensor(W.T, dtype=torch.float32))
        layer.bias.copy_(torch.tensor(b, dtype=torch.float32))

    X_t = torch.tensor(X, dtype=torch.float32)
    Y_t = layer(X_t)

    print("PyTorch layer.weight shape =", tuple(layer.weight.shape))
    print("PyTorch layer.bias shape =", tuple(layer.bias.shape))
    print("PyTorch 输出 =\n", Y_t.detach().numpy())
    print("与 NumPy X @ W + b 一致：", np.allclose(Y_t.detach().numpy(), Y))
    print("注意：PyTorch 存储 weight 为 (out_features, in_features)，")
    print("      因此前向等价于 X @ weight.T + bias。")
except Exception as e:
    print("未完成 PyTorch 对照：", repr(e))
    print("这不影响本讲 NumPy 主实验。")

print("\n" + "=" * 70)
print("实验结束")
print("=" * 70)
