"""第03讲实验：向量、点积与相似度
运行：python lesson03_dot_product_experiment.py
主线只依赖 NumPy；若安装了 PyTorch，会自动做一次结果核对。
"""
import numpy as np

np.set_printoptions(precision=4, suppress=True)


def manual_dot(a, b):
    """逐项相乘再求和，故意不用 np.dot。"""
    products = a * b
    return products, products.sum()


def cosine_similarity(a, b):
    """余弦相似度 = 点积 / (两个向量长度的乘积)。"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def show_pair(name, a, b):
    print(f"\n=== {name} ===")
    print("a =", a, "shape =", a.shape)
    print("b =", b, "shape =", b.shape)
    products, dot_manual = manual_dot(a, b)
    print("逐项乘积 a*b =", products)
    print("手工逻辑求和 =", dot_manual)
    print("np.dot(a,b) =", np.dot(a, b))
    print("|a| =", np.linalg.norm(a))
    print("|b| =", np.linalg.norm(b))
    print("cosine(a,b) =", cosine_similarity(a, b))


# 1) 最基本的数字例子
print("实验1：手算点积")
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])
show_pair("[1,2,3] 与 [4,5,6]", a, b)

# 2) 同方向、反方向、垂直：观察点积符号
print("\n实验2：方向如何影响点积")
q = np.array([1.0, 0.0])
show_pair("同方向", q, np.array([2.0, 0.0]))
show_pair("垂直", q, np.array([0.0, 3.0]))
show_pair("反方向", q, np.array([-2.0, 0.0]))

# 3) 尺度陷阱：点积大，不一定只是“方向更像”
print("\n实验3：点积会受到向量长度影响")
x = np.array([1.0, 1.0])
y_short = np.array([1.0, 1.0])
y_long = np.array([10.0, 10.0])
show_pair("x 与 y_short", x, y_short)
show_pair("x 与 y_long", x, y_long)

# 4) 一个微型 Attention 打分：1 个 query 对 3 个 key
print("\n实验4：微型 Q·K 打分")
query = np.array([1.0, 2.0])
keys = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
])
print("query shape =", query.shape)
print("keys shape =", keys.shape)
# 每一行 key 与 query 点积；矩阵写法等价于 keys @ query
scores_loop = np.array([np.dot(query, k) for k in keys])
scores_matrix = keys @ query
print("逐个点积 scores =", scores_loop)
print("矩阵乘法 scores =", scores_matrix)
print("scores shape =", scores_matrix.shape)
print("最大分数 key index =", int(np.argmax(scores_matrix)))

# 5) 可选：PyTorch 仅做结果核对；没有安装也不影响本讲
print("\n实验5（可选）：PyTorch 对照")
try:
    import torch
    ta = torch.tensor([1.0, 2.0, 3.0])
    tb = torch.tensor([4.0, 5.0, 6.0])
    print("torch.dot(ta,tb) =", torch.dot(ta, tb).item())
    tq = torch.tensor(query, dtype=torch.float32)
    tk = torch.tensor(keys, dtype=torch.float32)
    print("tk @ tq =", (tk @ tq))
except Exception as e:
    print("未运行 PyTorch 对照（这不影响本讲）：", type(e).__name__, str(e))

print("\n结论：点积是“对应维度相乘再求和”；它同时受到方向与向量长度影响。")
