# -*- coding: utf-8 -*-
"""
第51讲实验：为什么 Transformer 最终压过 RNN
目标：用最小实验观察长程依赖路径、Self-Attention 直接交互、因果 Mask、并行计算与 T^2 代价。
"""

import math  # 导入数学函数库，用于平方根等计算
import time  # 导入计时工具，用于粗略比较串行循环和矩阵运算
from pathlib import Path  # 用于安全地处理输出路径

import numpy as np  # 导入 NumPy，用于数组与绘图数据
import torch  # 导入 PyTorch，用于 Tensor 和矩阵计算
import matplotlib.pyplot as plt  # 导入 Matplotlib，用于生成实验图
from matplotlib.font_manager import FontProperties  # 导入中文字体属性

# 固定随机种子，保证每次运行尽量得到相同结果
torch.manual_seed(51)
np.random.seed(51)

# 定义输出目录
BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 指定系统中的中文字体，仅用于图中文字显示
CN_FONT = FontProperties(fname="/usr/share/fonts/truetype/arphic/uming.ttc")


def print_header(title):
    """打印实验小节标题。"""
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# -----------------------------------------------------------------------------
# 实验1：RNN 与 Self-Attention 的长程依赖路径长度
# -----------------------------------------------------------------------------
print_header("实验1：长程依赖的路径长度")

# 构造一组不同的序列长度
lengths = np.array([2, 4, 8, 16, 32, 64, 128, 256])

# 在最简单的单向 RNN 中，第1个 Token 到最后1个 Token 需要跨越 T-1 个递推边
rnn_path = lengths - 1

# 在单层全连接 Self-Attention 中，任意两个 Token 可以通过一次 Attention 直接建立联系
attention_path = np.ones_like(lengths)

# 打印一个具体例子
T_example = 128
print(f"序列长度 T={T_example}")
print(f"RNN：Token 1 -> Token {T_example} 的最短时间路径约为 {T_example - 1} 步")
print("Self-Attention：在一个 Attention Layer 内可直接建立联系，路径长度约为 1")

# 绘制路径长度对比图
plt.figure(figsize=(8, 4.8))
plt.plot(lengths, rnn_path, marker="o", label="RNN 时间路径 T-1")
plt.plot(lengths, attention_path, marker="s", label="Self-Attention 单层路径 1")
plt.xlabel("序列长度 T", fontproperties=CN_FONT)
plt.ylabel("最短依赖路径长度", fontproperties=CN_FONT)
plt.title("长程依赖：RNN 与 Self-Attention 的信息路径", fontproperties=CN_FONT)
plt.legend(prop=CN_FONT)
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(FIG_DIR / "lesson51_dependency_path_length.png", dpi=180)
plt.close()


# -----------------------------------------------------------------------------
# 实验2：手工构造 Self-Attention，观察远距离 Token 可以一步影响输出
# -----------------------------------------------------------------------------
print_header("实验2：Self-Attention 如何直接读取远距离 Token")

# 构造5个 Token 的 Value；为了直观，只让第1个 Token 携带明显信息
V = torch.tensor([
    [10.0, 0.0],  # Token 1 的 Value 很强
    [0.0, 1.0],
    [0.0, 2.0],
    [0.0, 3.0],
    [0.0, 4.0],   # Token 5 是当前 Query 所在位置
])

# 直接人为构造 Token 5 对所有 Token 的 attention scores
scores = torch.tensor([-1.0, -1.0, -1.0, -1.0, 4.0])

# 为了展示“远距离访问”，把 Token 1 的分数改成最高
scores_far = torch.tensor([5.0, -2.0, -2.0, -2.0, 0.0])

# 使用 Softmax 把 scores 变成 attention weights
weights_far = torch.softmax(scores_far, dim=-1)

# 用权重加权所有 Value，得到 Token 5 的新表示
output_far = weights_far @ V

print("Token 5 对各位置的 Attention Weights:")
print(weights_far)
print("Token 5 聚合后的输出:")
print(output_far)
print("说明：Token 5 可以在同一层中直接把大量权重放到远处的 Token 1。")

# 绘制 Attention 权重条形图
plt.figure(figsize=(7.5, 4.5))
plt.bar(np.arange(1, 6), weights_far.numpy())
plt.xlabel("被关注的 Token 位置", fontproperties=CN_FONT)
plt.ylabel("Attention Weight", fontproperties=CN_FONT)
plt.title("Token 5 可以直接关注远距离 Token 1", fontproperties=CN_FONT)
plt.xticks(np.arange(1, 6))
plt.tight_layout()
plt.savefig(FIG_DIR / "lesson51_direct_long_range_attention.png", dpi=180)
plt.close()


# -----------------------------------------------------------------------------
# 实验3：Causal Mask 允许训练时并行计算所有位置，但禁止看未来
# -----------------------------------------------------------------------------
print_header("实验3：Causal Mask：并行计算 != 可以偷看未来")

# 构造4个 Token、每个 Token 3维的表示
X = torch.tensor([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
])

# 为了简化，令 Q=K=V=X
Q = X.clone()
K = X.clone()
V2 = X.clone()

# 计算所有 Query-Key 分数；一次矩阵乘法就得到整个 T x T 矩阵
score_matrix = Q @ K.T / math.sqrt(Q.shape[-1])

# 构造上三角未来位置 Mask；对未来位置填入负无穷
future_mask = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
masked_scores = score_matrix.masked_fill(future_mask, float("-inf"))

# 对每一行做 Softmax，得到因果 Attention 权重
causal_weights = torch.softmax(masked_scores, dim=-1)

# 所有位置的输出仍然可以用一个矩阵乘法同时计算
causal_output = causal_weights @ V2

print("Causal Attention Weights:")
print(causal_weights)
print("每一行未来位置的权重都为0，但4个位置的训练计算可以同时完成。")
print("输出 Shape:", tuple(causal_output.shape))

# 绘制因果 Attention 矩阵
plt.figure(figsize=(5.2, 4.5))
plt.imshow(causal_weights.numpy(), aspect="auto")
plt.colorbar(label="Attention Weight")
plt.xlabel("Key / Value 位置", fontproperties=CN_FONT)
plt.ylabel("Query 位置", fontproperties=CN_FONT)
plt.title("Causal Mask 后的 Attention 权重", fontproperties=CN_FONT)
plt.xticks(range(4), [1, 2, 3, 4])
plt.yticks(range(4), [1, 2, 3, 4])
plt.tight_layout()
plt.savefig(FIG_DIR / "lesson51_causal_attention_matrix.png", dpi=180)
plt.close()


# -----------------------------------------------------------------------------
# 实验4：训练时并行计算的粗略演示（只用于结构直觉，不当作硬件基准）
# -----------------------------------------------------------------------------
print_header("实验4：串行递推 vs 一次矩阵投影的结构性并行差异")

# 设置一个中等规模的序列，用CPU做粗略演示
T = 1024
D = 128
H = 128

# 随机生成序列和权重
seq = torch.randn(T, D)
W_x = torch.randn(D, H) / math.sqrt(D)
W_h = torch.randn(H, H) / math.sqrt(H)
W_q = torch.randn(D, H) / math.sqrt(D)

# 预热矩阵计算，减少第一次调用的初始化影响
_ = seq @ W_q

# 计时：RNN 必须按时间步循环，因为 h_t 依赖 h_{t-1}
h = torch.zeros(H)
start = time.perf_counter()
for t in range(T):
    h = torch.tanh(seq[t] @ W_x + h @ W_h)
rnn_seconds = time.perf_counter() - start

# 计时：所有 Token 的 Q Projection 可以一次矩阵乘法完成
start = time.perf_counter()
Q_all = seq @ W_q
projection_seconds = time.perf_counter() - start

print(f"CPU 粗略计时：RNN Python 时间循环 = {rnn_seconds:.6f} s")
print(f"CPU 粗略计时：所有 Token 一次 Q Projection = {projection_seconds:.6f} s")
print("注意：这不是公平的端到端 RNN vs Transformer 性能 benchmark；只展示依赖结构允许的并行形式不同。")

# 绘制结构性计时图
plt.figure(figsize=(6.8, 4.4))
plt.bar(["RNN 时间循环", "一次矩阵投影"], [rnn_seconds, projection_seconds])
plt.ylabel("本机 CPU 秒数", fontproperties=CN_FONT)
plt.title("结构性并行演示（不是公平端到端 Benchmark）", fontproperties=CN_FONT)
plt.xticks(fontproperties=CN_FONT)
plt.tight_layout()
plt.savefig(FIG_DIR / "lesson51_parallelism_demo.png", dpi=180)
plt.close()


# -----------------------------------------------------------------------------
# 实验5：Self-Attention 的 T^2 代价
# -----------------------------------------------------------------------------
print_header("实验5：Self-Attention 为什么要付出 T^2 代价")

# 选择一系列序列长度
T_values = np.array([128, 256, 512, 1024, 2048, 4096, 8192])

# 一个 Attention Head 的 score matrix 元素数量约为 T^2
score_elements = T_values.astype(np.int64) ** 2

# 假设 fp16/bfloat16 每个 score 用2字节，仅估算原始 score matrix 大小
score_mib = score_elements * 2 / (1024 ** 2)

# 打印几个典型长度
for t, elems, mib in zip(T_values, score_elements, score_mib):
    print(f"T={t:5d}: T^2={elems:12d} 个 score，单 Head 原始 fp16 score 约 {mib:8.2f} MiB")

# 绘制二次增长曲线
plt.figure(figsize=(7.5, 4.8))
plt.plot(T_values, score_elements / 1e6, marker="o")
plt.xlabel("序列长度 T", fontproperties=CN_FONT)
plt.ylabel("单 Head Attention Score 元素数（百万）", fontproperties=CN_FONT)
plt.title("标准 Self-Attention 的 T² 增长", fontproperties=CN_FONT)
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(FIG_DIR / "lesson51_attention_quadratic_scaling.png", dpi=180)
plt.close()


# -----------------------------------------------------------------------------
# 实验6：路径长度与“生成仍然串行”的边界
# -----------------------------------------------------------------------------
print_header("实验6：Transformer 训练可并行，但自回归生成仍然逐 Token")

# 假设要生成8个新Token；第t个新Token必须在前t-1个新Token已经生成之后才能确定
new_tokens = 8
for t in range(1, new_tokens + 1):
    print(f"生成第 {t} 个新 Token 前，需要已有前 {t - 1} 个新 Token 的结果")

print("因此：Transformer 的优势主要体现在训练和层内计算的并行性；自回归解码时间轴仍然是串行的。")

print_header("实验完成")
print("已生成图文件：")
for path in sorted(FIG_DIR.glob("lesson51_*.png")):
    print(" -", path.name)
