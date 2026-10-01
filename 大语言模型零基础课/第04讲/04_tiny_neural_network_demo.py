"""
第4讲配套实验：不用 NumPy / PyTorch，手写一个两层小网络的前向计算。
目的：看清 x -> Wx+b -> ReLU -> 下一层。
"""

def matvec(W, x):
    """矩阵 W 乘向量 x。W 是“行”的列表。"""
    return [sum(w * xi for w, xi in zip(row, x)) for row in W]

def add_bias(v, b):
    return [vi + bi for vi, bi in zip(v, b)]

def relu(v):
    return [max(0.0, value) for value in v]

def linear(W, x, b):
    return add_bias(matvec(W, x), b)

# 输入向量：可以理解成某一层收到的 hidden state 的极小玩具版本
x = [1.0, 2.0]

# 第一层：3 个输出，因此 W1 有 3 行，每行对应 2 个输入权重
W1 = [
    [0.5, 1.0],
    [-1.0, 0.5],
    [2.0, -0.5],
]
b1 = [0.1, 0.0, -0.2]

# 第二层：把 3 维 hidden state 变成 2 维输出
W2 = [
    [1.0, -0.5, 0.2],
    [0.3, 0.8, -1.0],
]
b2 = [0.0, 0.2]

z1 = linear(W1, x, b1)
h1 = relu(z1)
y = linear(W2, h1, b2)

print("输入 x         =", x)
print("第一层 Wx+b z1 =", z1)
print("ReLU 后 h1     =", h1)
print("第二层输出 y   =", y)

print("\n参数数量：")
params_W1 = sum(len(row) for row in W1)
params_b1 = len(b1)
params_W2 = sum(len(row) for row in W2)
params_b2 = len(b2)
print("W1:", params_W1, "b1:", params_b1, "W2:", params_W2, "b2:", params_b2)
print("总参数:", params_W1 + params_b1 + params_W2 + params_b2)

print("\n动手修改建议：")
print("1) 把 W1[0][0] 从 0.5 改成 5.0，再运行一次。")
print("2) 把 h1 = relu(z1) 改成 h1 = z1，观察去掉非线性后的输出。")
