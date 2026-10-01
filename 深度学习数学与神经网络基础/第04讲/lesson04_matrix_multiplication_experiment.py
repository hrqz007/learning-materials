"""
第04讲实验：Matrix Multiplication（矩阵乘法）
目标：从手工计算理解矩阵乘法，再用 NumPy / PyTorch 验证，最后连接到 Transformer 的 Q/K/V 线性投影。

运行方式：
    python lesson04_matrix_multiplication_experiment.py
"""

import numpy as np

np.set_printoptions(precision=3, suppress=True)


def title(text):
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


def manual_matmul(A, B):
    """不用 np.matmul，按“行 dot 列”规则手写矩阵乘法。"""
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("manual_matmul 只处理二维矩阵。")
    if A.shape[1] != B.shape[0]:
        raise ValueError(
            f"Shape 不兼容：A{A.shape} 的列数 {A.shape[1]} != "
            f"B{B.shape} 的行数 {B.shape[0]}"
        )

    rows = A.shape[0]
    cols = B.shape[1]
    inner = A.shape[1]
    C = np.zeros((rows, cols), dtype=float)

    for i in range(rows):
        for j in range(cols):
            total = 0.0
            for k in range(inner):
                total += A[i, k] * B[k, j]
            C[i, j] = total
    return C


# ---------------------------------------------------------------------------
# 实验 1：2x3 乘 3x2，逐格理解“行 dot 列”
# ---------------------------------------------------------------------------
title("实验1：2x3 × 3x2 —— 每个输出格子都是一次 Dot Product")
A = np.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0]])
B = np.array([[7.0, 8.0],
              [9.0, 10.0],
              [11.0, 12.0]])

print("A =\n", A)
print("A.shape =", A.shape)
print("B =\n", B)
print("B.shape =", B.shape)

# 手工展示四个输出元素
c00 = 1*7 + 2*9 + 3*11
c01 = 1*8 + 2*10 + 3*12
c10 = 4*7 + 5*9 + 6*11
c11 = 4*8 + 5*10 + 6*12
print("\n逐格手算：")
print("C[0,0] = 1*7 + 2*9 + 3*11 =", c00)
print("C[0,1] = 1*8 + 2*10 + 3*12 =", c01)
print("C[1,0] = 4*7 + 5*9 + 6*11 =", c10)
print("C[1,1] = 4*8 + 5*10 + 6*12 =", c11)

C_manual = manual_matmul(A, B)
C_numpy = A @ B
print("\n手写 manual_matmul(A, B) =\n", C_manual)
print("NumPy A @ B =\n", C_numpy)
print("结果完全一致：", np.allclose(C_manual, C_numpy))
print("输出 Shape：", C_numpy.shape)


# ---------------------------------------------------------------------------
# 实验 2：Shape 规则 (m,n) @ (n,p) -> (m,p)
# ---------------------------------------------------------------------------
title("实验2：先预测 Shape，再计算")
X = np.arange(12, dtype=float).reshape(4, 3)
W = np.arange(15, dtype=float).reshape(3, 5)
Y = X @ W
print("X.shape =", X.shape)
print("W.shape =", W.shape)
print("预测： (4,3) @ (3,5) -> (4,5)")
print("实际 Y.shape =", Y.shape)
print("Y =\n", Y)


# ---------------------------------------------------------------------------
# 实验 3：故意制造 Shape 不匹配
# ---------------------------------------------------------------------------
title("实验3：为什么 (2,3) 不能直接乘 (2,4)")
Bad = np.ones((2, 4))
print("A.shape =", A.shape)
print("Bad.shape =", Bad.shape)
print("矩阵乘法要求：左矩阵的列数 == 右矩阵的行数")
try:
    _ = A @ Bad
except ValueError as e:
    print("捕获到预期错误：")
    print(str(e).split("\n")[0])


# ---------------------------------------------------------------------------
# 实验 4：矩阵乘法 vs 逐元素乘法
# ---------------------------------------------------------------------------
title("实验4：Matrix Multiplication != Element-wise Multiplication")
M1 = np.array([[1.0, 2.0],
               [3.0, 4.0]])
M2 = np.array([[5.0, 6.0],
               [7.0, 8.0]])
print("M1 * M2（逐元素乘法）=\n", M1 * M2)
print("M1 @ M2（矩阵乘法）=\n", M1 @ M2)


# ---------------------------------------------------------------------------
# 实验 5：神经网络的一层 —— 一批样本 X @ W
# ---------------------------------------------------------------------------
title("实验5：一个 Batch 经过线性权重矩阵")
# 3 个样本，每个样本 2 个输入特征
X_batch = np.array([[1.0, 2.0],
                    [0.5, -1.0],
                    [3.0, 0.0]])
# 2 个输入特征 -> 4 个输出特征
W_linear = np.array([[0.1, 0.2, 0.3, 0.4],
                     [1.0, -1.0, 0.5, 2.0]])
Y_linear = X_batch @ W_linear
print("X_batch.shape =", X_batch.shape, "  # [batch=3, input_dim=2]")
print("W_linear.shape =", W_linear.shape, " # [input_dim=2, output_dim=4]")
print("Y_linear.shape =", Y_linear.shape, " # [batch=3, output_dim=4]")
print("Y_linear =\n", Y_linear)
print("第1个样本的输出 =", Y_linear[0])
print("它等价于 [1,2] 分别和 W 的4个列向量做点积。")


# ---------------------------------------------------------------------------
# 实验 6：Transformer 中 Q = XW_Q, K = XW_K, V = XW_V
# ---------------------------------------------------------------------------
title("实验6：微型 Transformer 的 Q / K / V 线性投影")
# 4 个 token，每个 token 的 hidden size = 3
X_tokens = np.array([[1.0, 0.0, 1.0],
                     [0.0, 1.0, 1.0],
                     [1.0, 1.0, 0.0],
                     [0.5, 0.5, 0.5]])
W_Q = np.array([[1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0]])
W_K = np.array([[0.5, 0.0],
                [0.0, 0.5],
                [1.0, -1.0]])
W_V = np.array([[1.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0]])

Q = X_tokens @ W_Q
K = X_tokens @ W_K
V = X_tokens @ W_V

print("X_tokens.shape =", X_tokens.shape, "# [sequence=4, hidden=3]")
print("W_Q.shape =", W_Q.shape, "      # [hidden=3, q_dim=2]")
print("Q.shape =", Q.shape)
print("Q =\n", Q)
print("K.shape =", K.shape)
print("K =\n", K)
print("V.shape =", V.shape)
print("V =\n", V)
print("结论：同一个 X 乘不同的可学习权重矩阵，会得到不同的表示 Q/K/V。")


# ---------------------------------------------------------------------------
# 实验 7：可选 PyTorch 验证
# ---------------------------------------------------------------------------
title("实验7：PyTorch 验证（如果已安装 torch）")
try:
    import torch
    A_t = torch.tensor(A, dtype=torch.float32)
    B_t = torch.tensor(B, dtype=torch.float32)
    C_t = A_t @ B_t
    print("torch 版本：", torch.__version__)
    print("PyTorch A @ B =\n", C_t)
    print("与 NumPy 一致：", np.allclose(C_t.numpy(), C_numpy))
except Exception as e:
    print("PyTorch 跳过：", e)


title("实验总结")
print("1. (m,n) @ (n,p) -> (m,p)：中间维必须相等，外侧维被保留。")
print("2. 输出矩阵的每个元素，本质上都是：左矩阵的一行 dot 右矩阵的一列。")
print("3. 矩阵乘法不是逐元素乘法。")
print("4. 神经网络用 X @ W 一次处理一批样本和多个输出神经元。")
print("5. Transformer 的 Q=XW_Q、K=XW_K、V=XW_V 只是同一规则的大规模版本。")
