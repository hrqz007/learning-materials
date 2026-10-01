# 第06讲实验：Batch、批量矩阵运算与 LLM 的 (B, T, D)
# 目标：验证“同一组 Linear 参数可以同时作用于所有样本、所有 Token”。

import numpy as np

# 为了让打印出来的数字更易读，保留 3 位小数并关闭科学计数法。
np.set_printoptions(precision=3, suppress=True)

print("=" * 78)
print("实验1：二维 Batch 与逐样本计算完全等价")
print("=" * 78)

# X 有 3 个样本，每个样本 4 个输入特征，所以 shape = (B=3, D=4)。
X = np.array([[1.0, 2.0, 0.0, -1.0],
              [0.5, 0.0, 1.0,  2.0],
              [-1.0, 1.0, 2.0, 0.5]])

# W 把 4 维输入映射成 2 维输出，所以 shape = (D=4, O=2)。
W = np.array([[ 1.0,  0.5],
              [ 0.0, -1.0],
              [ 2.0,  0.0],
              [-0.5,  1.5]])

# 每个输出维度都有一个 bias，所以 shape = (O=2,)。
b = np.array([0.1, -0.2])

# 方法A：一次对整个 Batch 做矩阵乘法。
Y_batch = X @ W + b

# 方法B：逐个样本循环，使用同一套 W 和 b。
Y_loop = np.stack([X[i] @ W + b for i in range(X.shape[0])], axis=0)

print("X shape =", X.shape)
print("W shape =", W.shape)
print("b shape =", b.shape)
print("Y_batch shape =", Y_batch.shape)
print("Y_batch =\n", Y_batch)
print("逐样本结果与 Batch 一次计算一致：", np.allclose(Y_batch, Y_loop))

print("\n" + "=" * 78)
print("实验2：构造一个真正的 LLM 风格 Tensor：X.shape = (B, T, D)")
print("=" * 78)

# B=2：两个样本；T=3：每个样本 3 个 Token；D=4：每个 Token 4 维表示。
X3 = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4) / 10.0

# 仍然使用同一个 4 -> 2 的 Linear 参数。
Y3 = X3 @ W + b

print("X3 shape =", X3.shape, "  含义 = (B=2, T=3, D=4)")
print("W shape  =", W.shape,  "  含义 = (D=4, O=2)")
print("Y3 shape =", Y3.shape, "  预期 = (B=2, T=3, O=2)")
print("X3[0, 0] =", X3[0, 0])
print("Y3[0, 0] =", Y3[0, 0])
print("手工验证第一个 Token：", X3[0, 0] @ W + b)

print("\n" + "=" * 78)
print("实验3：三维直接 Linear == 先展平 B*T 再 Linear 再恢复")
print("=" * 78)

# 把前两个轴 B 和 T 合并，得到 6 个 Token，每个 Token 仍然是 4 维。
X_flat = X3.reshape(-1, X3.shape[-1])

# 对 6 个 Token 一次做相同 Linear。
Y_flat = X_flat @ W + b

# 恢复为 (B, T, O)。
Y_restore = Y_flat.reshape(X3.shape[0], X3.shape[1], W.shape[1])

print("X_flat shape =", X_flat.shape)
print("Y_flat shape =", Y_flat.shape)
print("Y_restore shape =", Y_restore.shape)
print("与直接 X3 @ W + b 完全一致：", np.allclose(Y3, Y_restore))

print("\n" + "=" * 78)
print("实验4：逐 Token 循环 == 三维直接矩阵乘法")
print("=" * 78)

# 创建与 Y3 同 shape 的空数组，用于保存逐 Token 计算结果。
Y_token_loop = np.empty_like(Y3)

# 第一层循环遍历 Batch。
for batch_index in range(X3.shape[0]):
    # 第二层循环遍历该样本中的每一个 Token。
    for token_index in range(X3.shape[1]):
        # 每个 Token 都使用完全相同的 W 和 b。
        Y_token_loop[batch_index, token_index] = X3[batch_index, token_index] @ W + b

print("逐 Token 循环与三维直接计算一致：", np.allclose(Y_token_loop, Y3))
print("这说明 Linear 并没有混合不同 Token；它只对最后一个 D 维做相同映射。")

print("\n" + "=" * 78)
print("实验5：Bias Broadcasting 在 (B,T,O) 上怎样发生")
print("=" * 78)

# 先只做矩阵乘法，不加 bias。
Z3 = X3 @ W

# 再加 shape=(2,) 的 b。NumPy 会把它广播到每个 Batch、每个 Token。
Y3_again = Z3 + b

print("Z3 shape =", Z3.shape)
print("b shape  =", b.shape)
print("Y3_again 与 Y3 一致：", np.allclose(Y3_again, Y3))
print("Y3_again - Z3 的第0个样本 =\n", Y3_again[0] - Z3[0])
print("可以看到每个 Token 都加上同一个 b =", b)

print("\n" + "=" * 78)
print("实验6：改变 Batch Size 或 Sequence Length，不会改变 Linear 参数数量")
print("=" * 78)

# 参数量只由输入维度 D 和输出维度 O 决定。
D = W.shape[0]
O = W.shape[1]
parameter_count = D * O + O

print("D =", D, ", O =", O)
print("Linear 参数量 = D*O + O =", parameter_count)
print("当前 B=2, T=3，参数量仍然是", parameter_count)
print("即使 B=128, T=4096，只要 D 和 O 不变，这一层参数量仍然不变。")

print("\n" + "=" * 78)
print("实验7：PyTorch nn.Linear 直接接收三维 Tensor（若已安装 PyTorch）")
print("=" * 78)

try:
    import torch
    import torch.nn as nn

    # 创建一个 4 -> 2 的 Linear 层。
    layer = nn.Linear(4, 2, bias=True)

    # 为了与 NumPy 完全对照，把 PyTorch 参数改成同样的 W 和 b。
    # PyTorch weight 的存储 shape 是 (out_features, in_features)，因此要复制 W.T。
    with torch.no_grad():
        layer.weight.copy_(torch.tensor(W.T, dtype=torch.float32))
        layer.bias.copy_(torch.tensor(b, dtype=torch.float32))

    # 把三维 NumPy Tensor 转成 PyTorch Tensor。
    X3_t = torch.tensor(X3, dtype=torch.float32)

    # nn.Linear 会自动作用于最后一维 D=4，并保留前面的 B 和 T 两个轴。
    Y3_t = layer(X3_t)

    print("PyTorch 输入 shape =", tuple(X3_t.shape))
    print("PyTorch weight shape =", tuple(layer.weight.shape))
    print("PyTorch 输出 shape =", tuple(Y3_t.shape))
    print("PyTorch 与 NumPy 结果一致：", np.allclose(Y3_t.detach().numpy(), Y3, atol=1e-6))
except Exception as e:
    print("PyTorch 对照未执行：", repr(e))
    print("这不影响前面的 NumPy 主实验。")

print("\n" + "=" * 78)
print("实验结束：本讲最重要的结论")
print("=" * 78)
print("1. Batch 不是给每个样本复制一套参数，而是让很多样本共享同一套参数并行计算。")
print("2. 对 (B,T,D) 做 Linear，本质上是对每个 Token 的最后一维 D 使用同一套 W、b。")
print("3. (B,T,D) -> Linear(D,O) -> (B,T,O)，前面的 B、T 轴被保留。")
print("4. Batch Size 和 Sequence Length 影响计算量/显存，但不直接改变这一层的参数数量。")
