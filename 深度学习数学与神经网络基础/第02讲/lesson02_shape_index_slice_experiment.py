# 第02讲实验代码：Shape、索引、切片、Reshape 与 Transpose
# 目标：先预测每一步 Shape，再运行代码验证。

import numpy as np


def show(name, array):
    """统一打印数组的名称、数值、ndim、shape 和 size。"""
    print("=" * 70)
    print(f"{name}")
    print("value =")
    print(array)
    print(f"ndim  = {array.ndim}")
    print(f"shape = {array.shape}")
    print(f"size  = {array.size}")


print("第02讲实验：Tensor 索引、切片、Reshape 与 Transpose")
print("请先在纸上预测每一步的 Shape，再对照程序输出。")

# ----------------------------------------------------------------------
# 1. 建立一个微型 LLM 风格的 3D Tensor
# 约定三个轴分别表示：[batch, sequence, hidden]
# batch = 2，sequence = 3，hidden = 4
# ----------------------------------------------------------------------
X = np.arange(24).reshape(2, 3, 4)
show("原始 X，语义为 [B,T,D] = [2,3,4]", X)

# ----------------------------------------------------------------------
# 2. 整数索引 vs 切片
# X[0] 使用整数索引，会把 batch 轴固定到一个位置，因此该轴消失。
# X[0:1] 使用切片，只把 batch 轴长度从 2 变成 1，轴仍然存在。
# ----------------------------------------------------------------------
x_index = X[0]
x_slice = X[0:1]
show("X[0]：取第0个样本，batch轴消失", x_index)
show("X[0:1]：取第0个样本，但保留batch轴", x_slice)

# ----------------------------------------------------------------------
# 3. 取所有样本的第1个 Token
# 整数索引 1 会删除 sequence 轴。
# 切片 1:2 会保留 sequence 轴，只是长度变成 1。
# ----------------------------------------------------------------------
token_index = X[:, 1, :]
token_slice = X[:, 1:2, :]
show("X[:, 1, :]：所有样本，第1个Token，sequence轴消失", token_index)
show("X[:, 1:2, :]：所有样本，第1个Token，保留sequence轴", token_slice)

# ----------------------------------------------------------------------
# 4. 取所有样本的最后一个 Token
# -1 表示最后一个位置。
# 结果保留 batch 和 hidden 两个轴。
# ----------------------------------------------------------------------
last_token = X[:, -1, :]
show("X[:, -1, :]：所有样本的最后一个Token", last_token)

# ----------------------------------------------------------------------
# 5. 只取某一个 hidden feature
# 整数索引会删除 hidden 轴；切片会保留 hidden 轴。
# ----------------------------------------------------------------------
hidden_index = X[:, :, 0]
hidden_slice = X[:, :, 0:1]
show("X[:, :, 0]：只取hidden feature 0，hidden轴消失", hidden_index)
show("X[:, :, 0:1]：只取hidden feature 0，但保留hidden轴", hidden_slice)

# ----------------------------------------------------------------------
# 6. Reshape：元素总数保持不变
# 原 X 有 2*3*4 = 24 个元素。
# ----------------------------------------------------------------------
reshape_6x4 = X.reshape(6, 4)
reshape_4x6 = X.reshape(4, 6)
reshape_auto = X.reshape(2, 3, -1)  # -1 由 NumPy 自动推断为 4
show("X.reshape(6, 4)", reshape_6x4)
show("X.reshape(4, 6)", reshape_4x6)
show("X.reshape(2, 3, -1)：自动推断最后一维", reshape_auto)

# ----------------------------------------------------------------------
# 7. 非法 reshape：目标 Shape 需要 32 个元素，但 X 只有 24 个。
# 用 try/except 捕获错误，便于观察实验现象而不中断程序。
# ----------------------------------------------------------------------
print("=" * 70)
print("尝试非法 reshape：X.reshape(2, 4, 4)")
try:
    bad = X.reshape(2, 4, 4)
    print("意外成功：", bad.shape)
except ValueError as error:
    print("按预期失败。NumPy 报错：")
    print(error)

# ----------------------------------------------------------------------
# 8. Transpose：交换轴顺序
# transpose(0, 2, 1) 表示新轴顺序使用原来的 axis 0、axis 2、axis 1。
# Shape 从 (2,3,4) 变成 (2,4,3)。
# ----------------------------------------------------------------------
transposed = X.transpose(0, 2, 1)
show("X.transpose(0, 2, 1)：[B,T,D] -> [B,D,T]", transposed)

# ----------------------------------------------------------------------
# 9. 对照实验：reshape 与 transpose 即使得到相同 Shape，值布局也不同
# ----------------------------------------------------------------------
A = np.array([[1, 2, 3],
              [4, 5, 6]])
A_reshape = A.reshape(3, 2)
A_transpose = A.T
show("原始 A", A)
show("A.reshape(3, 2)", A_reshape)
show("A.T（transpose）", A_transpose)
print("reshape 与 transpose 的 Shape 是否相同：", A_reshape.shape == A_transpose.shape)
print("reshape 与 transpose 的数值是否完全相同：", np.array_equal(A_reshape, A_transpose))

# ----------------------------------------------------------------------
# 10. 增加 batch 轴，再把它删除
# ----------------------------------------------------------------------
sample = X[0]  # Shape = (3,4)
sample_with_batch = np.expand_dims(sample, axis=0)  # Shape = (1,3,4)
restored = np.squeeze(sample_with_batch, axis=0)    # Shape = (3,4)
show("sample = X[0]", sample)
show("np.expand_dims(sample, axis=0)：增加batch轴", sample_with_batch)
show("np.squeeze(..., axis=0)：删除长度为1的batch轴", restored)
print("删除再恢复后，数值是否与原 sample 完全一致：", np.array_equal(sample, restored))

# ----------------------------------------------------------------------
# 11. 无参数 squeeze 的边界风险
# 所有长度为1的轴都会被删除。
# ----------------------------------------------------------------------
y = np.zeros((1, 3, 1, 4))
y_squeezed = np.squeeze(y)
show("y，Shape = (1,3,1,4)", y)
show("np.squeeze(y)：所有长度为1的轴都会删除", y_squeezed)

# ----------------------------------------------------------------------
# 12. 总结输出：把关键 Shape 集中打印，便于与实验设计文档核对
# ----------------------------------------------------------------------
print("=" * 70)
print("关键 Shape 汇总")
print(f"X.shape                  = {X.shape}")
print(f"X[0].shape               = {X[0].shape}")
print(f"X[0:1].shape             = {X[0:1].shape}")
print(f"X[:,1,:].shape           = {X[:,1,:].shape}")
print(f"X[:,1:2,:].shape         = {X[:,1:2,:].shape}")
print(f"X[:,-1,:].shape          = {X[:,-1,:].shape}")
print(f"X.reshape(6,4).shape     = {X.reshape(6,4).shape}")
print(f"X.transpose(0,2,1).shape = {X.transpose(0,2,1).shape}")
print("实验结束。请回到实验设计文档解释每一个 Shape 为什么这样变化。")
