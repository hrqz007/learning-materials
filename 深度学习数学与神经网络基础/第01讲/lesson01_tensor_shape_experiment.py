"""
《面向 LLM / Agent Systems 科研的深度学习零基础课》
第 01 讲实验：Scalar / Vector / Matrix / Tensor / Shape

目标：
1. 观察 ndim、shape、size 的区别。
2. 验证同一组元素可以拥有不同 Shape。
3. 用一个极小 3D Tensor 模拟 LLM hidden states 的 [batch, sequence, hidden] 结构。
"""

import numpy as np


def show_info(name, x):
    """打印一个 NumPy 数组最重要的结构信息。"""
    print("=" * 60)
    print("name :", name)          # 对象名称，便于区分不同实验对象
    print("value:\n", x)         # 打印真实数值，观察数字如何排列
    print("ndim :", x.ndim)       # 轴的个数
    print("shape:", x.shape)      # 每个轴的长度
    print("size :", x.size)       # 所有元素的总数
    print("dtype:", x.dtype)      # 每个元素的数据类型


print("第 01 讲实验：认识 Scalar / Vector / Matrix / Tensor / Shape")
print()

# -------------------------------------------------------------------
# 实验 1：创建 0D、1D、2D、3D 数据
# -------------------------------------------------------------------

# Scalar：只有一个数字，因此没有可继续索引的轴。
scalar = np.array(7.5)

# Vector：一排 3 个数字，因此只有 1 个轴，轴长度为 3。
vector = np.array([1, 2, 3])

# Matrix：2 行 3 列，因此有 2 个轴，Shape 为 (2, 3)。
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

# 3D Tensor：这里可以把它想象成 2 个矩阵叠在一起。
# Shape 是 (2, 2, 3)：
# 第 0 轴长度 2；第 1 轴长度 2；第 2 轴长度 3。
tensor3d = np.array([
    [
        [1, 2, 3],
        [4, 5, 6]
    ],
    [
        [7, 8, 9],
        [10, 11, 12]
    ]
])

objects = [
    ("scalar", scalar),
    ("vector", vector),
    ("matrix", matrix),
    ("tensor3d", tensor3d),
]

for name, x in objects:
    show_info(name, x)

# -------------------------------------------------------------------
# 实验 2：相同的 6 个元素，换不同 Shape
# -------------------------------------------------------------------

print("\n" + "#" * 60)
print("实验 2：相同 6 个元素，不同 Shape")

# 先创建一个长度为 6 的向量。
base = np.array([1, 2, 3, 4, 5, 6])

# reshape 只改变这些数字的组织形状，不改变元素总数。
# reshape 会在第 02 讲正式学习，本讲只把它当作实验工具。
as_2x3 = base.reshape(2, 3)
as_3x2 = base.reshape(3, 2)

show_info("base_(6,)", base)
show_info("as_2x3", as_2x3)
show_info("as_3x2", as_3x2)

print("\n观察：三个对象的 size 都是 6，但 shape 不同。")

# -------------------------------------------------------------------
# 实验 3：模拟一个微型 LLM hidden states Tensor
# -------------------------------------------------------------------

print("\n" + "#" * 60)
print("实验 3：模拟 LLM 的 [batch, sequence, hidden]")

# np.arange(24) 先生成 0~23 共 24 个数字。
# 然后把它们组织成 Shape = (2, 3, 4)。
llm_hidden = np.arange(24).reshape(2, 3, 4)

# 我们人为规定三个轴的语义：
# axis 0 = batch：一次有 2 个样本
# axis 1 = sequence：每个样本有 3 个 Token
# axis 2 = hidden：每个 Token 有 4 个隐藏特征
show_info("llm_hidden", llm_hidden)

print("\n轴语义：")
print("axis 0 -> batch    ->", llm_hidden.shape[0])
print("axis 1 -> sequence ->", llm_hidden.shape[1])
print("axis 2 -> hidden   ->", llm_hidden.shape[2])

# 取出第 0 个样本、第 1 个 Token 的完整 hidden vector。
token_hidden = llm_hidden[0, 1]
print("\n第 0 个样本的第 1 个 Token 的 hidden vector:")
print(token_hidden)
print("它的 Shape 是:", token_hidden.shape)

# -------------------------------------------------------------------
# 最后总结
# -------------------------------------------------------------------

print("\n" + "=" * 60)
print("实验总结")
print("1. ndim = 轴的个数")
print("2. shape = 每个轴的长度")
print("3. size = 所有元素的总数")
print("4. 相同元素可以拥有不同 Shape")
print("5. LLM 中常见的 3D Tensor 可以理解为 [batch, sequence, hidden]")
