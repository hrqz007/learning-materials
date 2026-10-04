# -*- coding: utf-8 -*-
"""
第45讲实验：为什么需要 Normalization
目标：通过可复现实验观察均值/方差、特征尺度、深层激活漂移、激活饱和和 LayerNorm 预览。
"""

# 导入 pathlib，用来创建结果图片目录。
from pathlib import Path
# 导入 NumPy，用于手工数值实验。
import numpy as np
# 导入 PyTorch，用于和真实深度学习框架对照。
import torch
# 导入 matplotlib，用于生成实验图。
import matplotlib.pyplot as plt
# 导入字体管理模块，用于指定中文字体。
from matplotlib import font_manager

# 固定 NumPy 随机种子，保证实验可复现。
np.random.seed(45)
# 固定 PyTorch 随机种子，保证实验可复现。
torch.manual_seed(45)

# 获得当前脚本所在目录。
ROOT = Path(__file__).resolve().parent
# 创建保存图片的目录。
FIG_DIR = ROOT / "figures"
# 如果目录不存在就自动创建。
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 指定容器中可用的中文字体，避免图片中文乱码。
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
# 如果字体文件存在，就注册并使用该字体。
if Path(FONT_PATH).exists():
    # 把字体加入 matplotlib 字体管理器。
    font_manager.fontManager.addfont(FONT_PATH)
    # 设置全局字体为 Noto Sans CJK SC。
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
# 保证坐标轴负号正常显示。
plt.rcParams["axes.unicode_minus"] = False

# 定义一个标题打印函数，让文本输出更容易阅读。
def section(title: str) -> None:
    # 打印分隔线。
    print("\n" + "=" * 72)
    # 打印当前实验标题。
    print(title)
    # 再打印一条分隔线。
    print("=" * 72)

# 定义“按最后一个维度标准化”的 NumPy 函数。
def standardize_last_dim(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    # 计算最后一个维度的均值，并保留维度方便广播。
    mean = x.mean(axis=-1, keepdims=True)
    # 计算最后一个维度的总体方差，并保留维度方便广播。
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    # 用“减均值 / 标准差”完成标准化，并加入 eps 避免除零。
    return (x - mean) / np.sqrt(var + eps)

# -------------------- 实验1：最小数字例子 --------------------
section("实验1：标准化到底把一组数字变成什么？")
# 定义一个最小的一维向量。
x = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float64)
# 计算这个向量的均值。
mean = x.mean()
# 计算这个向量的总体方差。
var = ((x - mean) ** 2).mean()
# 计算这个向量的总体标准差。
std = np.sqrt(var)
# 对向量做标准化。
x_norm = (x - mean) / np.sqrt(var + 1e-5)
# 构造一个“整体放大10倍再平移100”的版本。
x_shift_scale = 10.0 * x + 100.0
# 对放大和平移后的向量再次做标准化。
x_shift_scale_norm = standardize_last_dim(x_shift_scale)
# 打印原始向量。
print("原始 x:", x)
# 打印均值、方差和标准差。
print(f"mean={mean:.6f}, var={var:.6f}, std={std:.6f}")
# 打印标准化结果。
print("standardized(x):", np.round(x_norm, 6))
# 打印标准化后的均值。
print(f"标准化后 mean={x_norm.mean():.8f}")
# 打印标准化后的总体标准差。
print(f"标准化后 std(population)={x_norm.std(ddof=0):.8f}")
# 打印放大和平移后的原始数据。
print("10*x+100:", x_shift_scale)
# 打印放大和平移后再标准化的结果。
print("standardized(10*x+100):", np.round(x_shift_scale_norm, 6))
# 打印两个标准化结果的最大绝对差异。
print("两种标准化结果最大差异:", float(np.max(np.abs(x_norm - x_shift_scale_norm))))

# 创建一个柱状图，展示原始值与标准化值。
fig, ax = plt.subplots(figsize=(8, 4.8))
# 创建横坐标位置。
idx = np.arange(len(x))
# 画原始值柱子。
ax.bar(idx - 0.2, x, width=0.4, label="原始 x")
# 画标准化值柱子。
ax.bar(idx + 0.2, x_norm, width=0.4, label="标准化后")
# 设置横轴刻度。
ax.set_xticks(idx)
# 设置横轴标签。
ax.set_xlabel("元素位置")
# 设置纵轴标签。
ax.set_ylabel("数值")
# 设置标题。
ax.set_title("实验1：标准化会改变数值尺度，但保留相对结构")
# 显示图例。
ax.legend()
# 自动调整排版。
fig.tight_layout()
# 保存图片。
fig.savefig(FIG_DIR / "lesson45_standardization_numbers.png", dpi=180)
# 关闭图片对象释放内存。
plt.close(fig)

# -------------------- 实验2：特征尺度与梯度 --------------------
section("实验2：Feature Scale 为什么会直接影响 Gradient Scale？")
# 构造三个量级完全不同的特征。
features = np.array([1.0, 100.0, 10000.0], dtype=np.float64)
# 把三个权重都初始化为0，便于手算。
weights = np.zeros_like(features)
# 设置目标值为1。
target = 1.0
# 计算当前线性模型预测值。
prediction = float(features @ weights)
# 计算平方误差 L=(pred-target)^2 对权重的梯度。
grad_raw = 2.0 * (prediction - target) * features
# 定义每个特征的典型尺度。
feature_scale = np.array([1.0, 100.0, 10000.0])
# 用典型尺度把三个输入压到同一数量级。
features_scaled = features / feature_scale
# 重新计算缩放后模型在零权重下的预测。
prediction_scaled = float(features_scaled @ weights)
# 计算缩放输入后的权重梯度。
grad_scaled = 2.0 * (prediction_scaled - target) * features_scaled
# 打印原始特征。
print("原始 features:", features)
# 打印原始梯度。
print("原始 dL/dw:", grad_raw)
# 打印缩放后的特征。
print("缩放后 features:", features_scaled)
# 打印缩放后的梯度。
print("缩放后 dL/dw:", grad_scaled)
# 打印最大与最小梯度绝对值的比例。
print("原始梯度绝对值最大/最小比例:", float(np.max(np.abs(grad_raw)) / np.min(np.abs(grad_raw))))

# 创建梯度尺度对比图。
fig, ax = plt.subplots(figsize=(8, 4.8))
# 画原始梯度绝对值。
ax.bar(idx[:3] - 0.18, np.abs(grad_raw), width=0.36, label="原始特征")
# 画缩放后梯度绝对值。
ax.bar(idx[:3] + 0.18, np.abs(grad_scaled), width=0.36, label="缩放后特征")
# 使用对数纵轴，因为原始梯度跨越多个数量级。
ax.set_yscale("log")
# 设置横轴刻度。
ax.set_xticks(idx[:3])
# 设置横轴标签文本。
ax.set_xticklabels(["feature 1", "feature 2", "feature 3"])
# 设置纵轴标签。
ax.set_ylabel("|dL/dw|（对数坐标）")
# 设置标题。
ax.set_title("实验2：Feature Scale 会直接改变 Gradient Scale")
# 显示图例。
ax.legend()
# 自动调整排版。
fig.tight_layout()
# 保存图片。
fig.savefig(FIG_DIR / "lesson45_feature_scale_gradient.png", dpi=180)
# 关闭图片对象。
plt.close(fig)

# -------------------- 实验3：深层激活尺度漂移 --------------------
section("实验3：深层网络里 Activation 的均值和方差怎样逐层漂移？")
# 创建独立随机数生成器，保证这一实验固定可复现。
rng = np.random.default_rng(45)
# 设置 batch 大小。
B = 512
# 设置隐藏维度。
D = 64
# 设置网络层数。
DEPTH = 20
# 生成初始输入，均值约0、标准差约1。
h0 = rng.normal(0.0, 1.0, size=(B, D))
# 创建保存每层权重和偏置的列表。
layers = []
# 按固定尺度生成20层权重。
for _ in range(DEPTH):
    # 故意使用稍偏大的权重尺度，让深层尺度漂移更容易观察。
    W = rng.normal(0.0, 1.20 * np.sqrt(2.0 / D), size=(D, D))
    # 给每一层一个小正偏置，使均值漂移也可观察。
    b = np.full((D,), 0.05, dtype=np.float64)
    # 保存这一层参数。
    layers.append((W, b))

# 定义一个函数，运行深层网络并记录每层激活统计量。
def run_deep_stack(use_conceptual_norm: bool):
    # 从同一个初始输入开始。
    h = h0.copy()
    # 保存每层均值。
    means = []
    # 保存每层标准差。
    stds = []
    # 逐层执行 Linear + （可选标准化）+ ReLU。
    for W, b in layers:
        # 执行线性变换。
        z = h @ W + b
        # 如果开启概念性标准化，就在每个样本的最后一个特征维上标准化。
        if use_conceptual_norm:
            # 这里只是为了说明“稳定尺度”的概念，不把它当成完整 LayerNorm 教程。
            z = standardize_last_dim(z)
        # 使用 ReLU 得到下一层激活。
        h = np.maximum(z, 0.0)
        # 记录当前激活的整体均值。
        means.append(float(h.mean()))
        # 记录当前激活的整体标准差。
        stds.append(float(h.std()))
    # 返回统计结果。
    return np.array(means), np.array(stds)

# 运行不做标准化的深层网络。
mean_raw, std_raw = run_deep_stack(False)
# 运行加入“概念性按特征标准化”的深层网络。
mean_norm, std_norm = run_deep_stack(True)
# 打印无标准化时的关键层统计量。
print("无标准化：第1/5/10/20层 mean =", np.round(mean_raw[[0, 4, 9, 19]], 6))
# 打印无标准化时的关键层标准差。
print("无标准化：第1/5/10/20层 std  =", np.round(std_raw[[0, 4, 9, 19]], 6))
# 打印有概念性标准化时的关键层统计量。
print("概念性标准化：第1/5/10/20层 mean =", np.round(mean_norm[[0, 4, 9, 19]], 6))
# 打印有概念性标准化时的关键层标准差。
print("概念性标准化：第1/5/10/20层 std  =", np.round(std_norm[[0, 4, 9, 19]], 6))

# 创建激活标准差随深度变化图。
fig, ax = plt.subplots(figsize=(8.5, 4.8))
# 创建层编号。
layer_idx = np.arange(1, DEPTH + 1)
# 画无标准化曲线。
ax.plot(layer_idx, std_raw, marker="o", markersize=3, label="无标准化")
# 画概念性标准化曲线。
ax.plot(layer_idx, std_norm, marker="o", markersize=3, label="逐层概念性标准化")
# 设置横轴标签。
ax.set_xlabel("层数")
# 设置纵轴标签。
ax.set_ylabel("Activation Std")
# 设置标题。
ax.set_title("实验3：深层网络中 Activation Scale 的逐层漂移")
# 显示图例。
ax.legend()
# 自动调整排版。
fig.tight_layout()
# 保存图片。
fig.savefig(FIG_DIR / "lesson45_deep_activation_scale.png", dpi=180)
# 关闭图片对象。
plt.close(fig)

# -------------------- 实验4：激活函数饱和 --------------------
section("实验4：输入尺度过大为什么会让 Sigmoid / Tanh 更容易进入饱和区？")
# 定义需要测试的输入标准差。
input_stds = np.array([0.5, 1.0, 2.0, 5.0, 10.0], dtype=np.float64)
# 创建保存 Sigmoid 饱和比例的列表。
sigmoid_saturated = []
# 创建保存 Tanh 饱和比例的列表。
tanh_saturated = []
# 逐个测试不同输入尺度。
for s in input_stds:
    # 生成大量正态分布输入。
    z = rng.normal(0.0, s, size=200000)
    # 计算 Sigmoid。
    sig = 1.0 / (1.0 + np.exp(-z))
    # 计算 Sigmoid 导数。
    sig_grad = sig * (1.0 - sig)
    # 把导数小于0.01视为明显饱和，统计比例。
    sigmoid_saturated.append(float(np.mean(sig_grad < 0.01)))
    # 计算 Tanh。
    th = np.tanh(z)
    # 计算 Tanh 导数。
    th_grad = 1.0 - th ** 2
    # 把导数小于0.01视为明显饱和，统计比例。
    tanh_saturated.append(float(np.mean(th_grad < 0.01)))
# 把列表转成数组方便后续画图。
sigmoid_saturated = np.array(sigmoid_saturated)
# 把列表转成数组方便后续画图。
tanh_saturated = np.array(tanh_saturated)
# 打印 Sigmoid 饱和比例。
print("Sigmoid 饱和比例:", np.round(sigmoid_saturated, 4))
# 打印 Tanh 饱和比例。
print("Tanh 饱和比例:", np.round(tanh_saturated, 4))

# 创建饱和比例曲线图。
fig, ax = plt.subplots(figsize=(8, 4.8))
# 画 Sigmoid 饱和比例。
ax.plot(input_stds, sigmoid_saturated, marker="o", label="Sigmoid: derivative < 0.01")
# 画 Tanh 饱和比例。
ax.plot(input_stds, tanh_saturated, marker="o", label="Tanh: derivative < 0.01")
# 设置横轴标签。
ax.set_xlabel("输入 z 的标准差")
# 设置纵轴标签。
ax.set_ylabel("饱和样本比例")
# 设置纵轴范围。
ax.set_ylim(0.0, 1.0)
# 设置标题。
ax.set_title("实验4：输入尺度越大，饱和激活比例越高")
# 显示图例。
ax.legend()
# 自动调整排版。
fig.tight_layout()
# 保存图片。
fig.savefig(FIG_DIR / "lesson45_activation_saturation.png", dpi=180)
# 关闭图片对象。
plt.close(fig)

# -------------------- 实验5：LayerNorm 预览 --------------------
section("实验5：用 PyTorch LayerNorm 预览“最后一个特征维归一化”")
# 创建一个 LLM 风格 Tensor，Shape 为 [B,T,D]。
X = torch.tensor(
    [
        [[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]],
        [[-2.0, 0.0, 2.0, 4.0], [100.0, 110.0, 120.0, 130.0]],
    ],
    dtype=torch.float32,
)
# 创建 LayerNorm；这里只做第47讲前的功能预览。
ln = torch.nn.LayerNorm(4, elementwise_affine=False)
# 对最后一个 hidden 维执行 LayerNorm。
Y = ln(X)
# 计算每个 token 在 hidden 维上的均值。
mean_per_token = Y.mean(dim=-1)
# 计算每个 token 在 hidden 维上的总体标准差。
std_per_token = Y.std(dim=-1, unbiased=False)
# 打印输入 Shape。
print("X.shape:", tuple(X.shape))
# 打印输出 Shape。
print("Y.shape:", tuple(Y.shape))
# 打印每个 token 归一化后的均值。
print("每个 token 的 hidden mean:\n", mean_per_token)
# 打印每个 token 归一化后的总体标准差。
print("每个 token 的 hidden std(population):\n", std_per_token)
# 打印一个 token 的输入。
print("示例 token 输入:", X[0, 1])
# 打印同一个 token 的输出。
print("示例 token 输出:", torch.round(Y[0, 1] * 10000) / 10000)

# 创建一张图，对比两个尺度差异很大的 token 在 LayerNorm 后的数值。
fig, ax = plt.subplots(figsize=(8, 4.8))
# 创建 hidden 维位置。
hidden_idx = np.arange(4)
# 画输入 [1,2,3,4] 归一化后的结果。
ax.plot(hidden_idx, Y[0, 0].numpy(), marker="o", label="token [1,2,3,4] -> LN")
# 画输入 [10,20,30,40] 归一化后的结果。
ax.plot(hidden_idx, Y[0, 1].numpy(), marker="s", label="token [10,20,30,40] -> LN")
# 设置横轴刻度。
ax.set_xticks(hidden_idx)
# 设置横轴标签。
ax.set_xlabel("hidden feature index")
# 设置纵轴标签。
ax.set_ylabel("归一化后的值")
# 设置标题。
ax.set_title("实验5：整体放大后的 token 经过 LayerNorm 后得到相同标准化形状")
# 显示图例。
ax.legend()
# 自动调整排版。
fig.tight_layout()
# 保存图片。
fig.savefig(FIG_DIR / "lesson45_layernorm_preview.png", dpi=180)
# 关闭图片对象。
plt.close(fig)

# 打印实验结束信息。
section("实验完成")
# 告诉用户结果图片保存位置。
print("所有实验图已保存到:", FIG_DIR)
