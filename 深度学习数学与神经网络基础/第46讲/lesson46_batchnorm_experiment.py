# 第46讲：BatchNorm 实验代码
# 目标：从数字、Shape、train/eval、running statistics 和 batch dependence 五个角度理解 BatchNorm。

# 导入 NumPy，用于数值计算和重复抽样实验。
import numpy as np
# 导入 PyTorch。
import torch
# 导入神经网络模块。
import torch.nn as nn
# 导入 Matplotlib，用于保存实验图。
import matplotlib.pyplot as plt
# 导入 Matplotlib 字体管理工具。
from matplotlib import font_manager
# 指定容器里实际存在的中文字体文件。
CJK_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
# 把该字体注册到 Matplotlib。
font_manager.fontManager.addfont(CJK_FONT_PATH)
# 读取字体在 Matplotlib 中的真实名称。
CJK_FONT_NAME = font_manager.FontProperties(fname=CJK_FONT_PATH).get_name()
# 设置全局默认字体为支持中文的 CJK 字体。
plt.rcParams["font.family"] = CJK_FONT_NAME
# 正常显示数学坐标轴中的负号。
plt.rcParams["axes.unicode_minus"] = False
# 导入 Path，便于创建输出目录。
from pathlib import Path

# 固定 NumPy 随机种子，保证重复运行尽量得到相同结果。
np.random.seed(46)
# 固定 PyTorch 随机种子，保证随机 Tensor 可复现。
torch.manual_seed(46)

# 找到当前脚本所在目录。
ROOT = Path(__file__).resolve().parent
# 定义图片输出目录。
FIG_DIR = ROOT / "figures"
# 如果图片目录不存在，就自动创建。
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 设置 NumPy 的打印格式，让输出更容易阅读。
np.set_printoptions(precision=6, suppress=True)
# 设置 PyTorch 的打印精度。
torch.set_printoptions(precision=6, sci_mode=False)

# 定义一个标题打印函数，方便把不同实验分隔开。
def title(text):
    # 打印一条分隔线。
    print("\n" + "=" * 78)
    # 打印实验标题。
    print(text)
    # 再打印一条分隔线。
    print("=" * 78)


# -----------------------------------------------------------------------------
# 实验1：手工 BatchNorm 与 nn.BatchNorm1d 对齐
# -----------------------------------------------------------------------------
title("实验1：手工 BatchNorm 与 nn.BatchNorm1d 对齐")

# 构造 4 个样本、3 个特征的输入矩阵，Shape = (N, C) = (4, 3)。
X = torch.tensor(
    [
        [1.0, 10.0, 100.0],
        [2.0, 12.0, 90.0],
        [3.0, 14.0, 110.0],
        [4.0, 16.0, 100.0],
    ],
    dtype=torch.float32,
)
# 定义数值稳定项 epsilon。
eps = 1e-5
# 沿 Batch 轴 dim=0 计算每个 Feature / Channel 的均值。
batch_mean = X.mean(dim=0)
# 沿 Batch 轴 dim=0 计算总体方差；unbiased=False 对应 Forward 中使用的 batch variance。
batch_var = X.var(dim=0, unbiased=False)
# 按 BatchNorm 公式手工标准化。
X_hat_manual = (X - batch_mean) / torch.sqrt(batch_var + eps)

# 创建 BatchNorm1d；关闭 affine，避免 gamma / beta 干扰；关闭 running stats，便于只看当前 batch。
bn = nn.BatchNorm1d(num_features=3, eps=eps, affine=False, track_running_stats=False)
# 切换到训练模式。
bn.train()
# 用 PyTorch BatchNorm 计算输出。
X_hat_torch = bn(X)
# 计算手工结果与 PyTorch 结果的最大绝对误差。
max_diff = (X_hat_manual - X_hat_torch).abs().max().item()

# 打印输入 Shape。
print("X.shape =", tuple(X.shape))
# 打印每个 Feature 的 Batch Mean。
print("batch mean =", batch_mean)
# 打印每个 Feature 的 Batch Variance（biased / population form）。
print("batch var  =", batch_var)
# 打印手工标准化结果。
print("manual normalized =\n", X_hat_manual)
# 打印 PyTorch BatchNorm 结果。
print("torch normalized  =\n", X_hat_torch)
# 打印最大绝对误差。
print(f"manual vs torch max abs diff = {max_diff:.10f}")
# 打印标准化后每个 Feature 的均值。
print("normalized mean =", X_hat_torch.mean(dim=0))
# 打印标准化后每个 Feature 的总体标准差。
print("normalized std  =", X_hat_torch.std(dim=0, unbiased=False))

# 保存标准化前各 Feature 的均值图。
plt.figure(figsize=(7, 4.5))
plt.bar(np.arange(3), batch_mean.numpy())
plt.xticks(np.arange(3), ["feature 0", "feature 1", "feature 2"])
plt.ylabel("Batch mean")
plt.title("BatchNorm 前：不同 Feature 的 Batch Mean")
plt.tight_layout()
plt.savefig(FIG_DIR / "lesson46_feature_mean_before.png", dpi=180)
plt.close()

# 保存标准化前后标准差对照图。
std_before = X.std(dim=0, unbiased=False).numpy()
std_after = X_hat_torch.std(dim=0, unbiased=False).numpy()
indices = np.arange(3)
width = 0.35
plt.figure(figsize=(7, 4.5))
plt.bar(indices - width / 2, std_before, width, label="before")
plt.bar(indices + width / 2, std_after, width, label="after")
plt.xticks(indices, ["feature 0", "feature 1", "feature 2"])
plt.ylabel("Standard deviation")
plt.title("BatchNorm 前后：每个 Feature 的尺度")
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "lesson46_feature_std_before_after.png", dpi=180)
plt.close()


# -----------------------------------------------------------------------------
# 实验2：gamma / beta 为什么存在
# -----------------------------------------------------------------------------
title("实验2：可学习 gamma / beta 为什么存在")

# 创建带 affine=True 的 BatchNorm1d，因此它内部有可学习 gamma(weight) 和 beta(bias)。
bn_affine = nn.BatchNorm1d(3, eps=eps, affine=True, track_running_stats=False)
# 关闭梯度记录，只是为了手动写入固定参数。
with torch.no_grad():
    # 设置 gamma。
    bn_affine.weight.copy_(torch.tensor([2.0, 0.5, 1.5]))
    # 设置 beta。
    bn_affine.bias.copy_(torch.tensor([1.0, -1.0, 0.2]))
# 切换训练模式。
bn_affine.train()
# 计算 BatchNorm 输出。
Y_affine = bn_affine(X)
# 按公式 y = gamma * x_hat + beta 手工计算。
Y_affine_manual = X_hat_manual * bn_affine.weight + bn_affine.bias
# 计算误差。
affine_diff = (Y_affine - Y_affine_manual).abs().max().item()

# 打印 gamma。
print("gamma =", bn_affine.weight.detach())
# 打印 beta。
print("beta  =", bn_affine.bias.detach())
# 打印输出 Shape。
print("output shape =", tuple(Y_affine.shape))
# 打印最大误差。
print(f"affine manual vs torch max abs diff = {affine_diff:.10f}")
# 打印输出均值，说明 gamma/beta 可以重新移动尺度与中心。
print("output mean after gamma/beta =", Y_affine.mean(dim=0).detach())
# 打印输出总体标准差。
print("output std after gamma/beta  =", Y_affine.std(dim=0, unbiased=False).detach())


# -----------------------------------------------------------------------------
# 实验3：running_mean / running_var 与 train() / eval()
# -----------------------------------------------------------------------------
title("实验3：running statistics 与 train/eval 差异")

# 创建 BatchNorm1d；关闭 affine，专注观察 running statistics。
bn_running = nn.BatchNorm1d(3, affine=False, momentum=0.2, track_running_stats=True)
# 进入训练模式。
bn_running.train()
# 准备记录每一步 running mean。
running_means = []
# 准备记录每个 batch 的均值。
batch_means = []

# 连续喂入 6 个均值逐渐上移的 batch。
for step in range(6):
    # 构造一个 8x3 的 batch；随着 step 增加，整体均值逐渐变大。
    batch = torch.randn(8, 3) * 1.5 + float(step)
    # 记录当前 batch 均值。
    current_mean = batch.mean(dim=0)
    # 前向一次；训练模式会使用当前 batch 统计量，并更新 running statistics。
    _ = bn_running(batch)
    # 记录当前 batch mean。
    batch_means.append(current_mean.detach().numpy().copy())
    # 记录更新后的 running mean。
    running_means.append(bn_running.running_mean.detach().numpy().copy())
    # 打印关键统计量。
    print(
        f"step={step:02d} | batch_mean={current_mean.numpy()} | "
        f"running_mean={bn_running.running_mean.numpy()}"
    )

# 转为 NumPy 数组便于作图。
running_means_np = np.asarray(running_means)
# 转为 NumPy 数组便于作图。
batch_means_np = np.asarray(batch_means)

# 绘制第 0 个 Feature 的 batch mean 与 running mean。
plt.figure(figsize=(7, 4.5))
plt.plot(batch_means_np[:, 0], marker="o", label="current batch mean")
plt.plot(running_means_np[:, 0], marker="o", label="running mean")
plt.xlabel("Training batch index")
plt.ylabel("Feature 0 mean")
plt.title("Batch Mean 与 Running Mean：running statistics 会缓慢跟踪")
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "lesson46_running_mean_evolution.png", dpi=180)
plt.close()

# 准备一个固定样本，用来比较 train / eval。
probe = torch.tensor([[1.0, 2.0, 3.0]])
# 切换到评估模式。
bn_running.eval()
# 评估模式用 running statistics，而不是当前 probe 自己的 batch statistics。
probe_eval = bn_running(probe)
# 手工使用 running mean / running var 计算评估输出。
probe_eval_manual = (probe - bn_running.running_mean) / torch.sqrt(bn_running.running_var + bn_running.eps)
# 打印 running mean。
print("final running_mean =", bn_running.running_mean)
# 打印 running var。
print("final running_var  =", bn_running.running_var)
# 打印 eval 输出。
print("eval(probe) =", probe_eval)
# 打印手工 eval 输出。
print("manual eval(probe) =", probe_eval_manual)
# 打印误差。
print(
    "eval manual vs torch max abs diff =",
    float((probe_eval - probe_eval_manual).abs().max()),
)


# -----------------------------------------------------------------------------
# 实验4：同一个样本，在训练态会依赖 batch 同伴
# -----------------------------------------------------------------------------
title("实验4：同一个样本在训练态依赖 batch 同伴")

# 创建 BatchNorm1d；训练态使用 batch statistics。
bn_dep = nn.BatchNorm1d(3, affine=False, track_running_stats=True)
# 手工把 running mean 设置为 0，便于评估态对比。
with torch.no_grad():
    # 设置 running mean 为 0。
    bn_dep.running_mean.zero_()
    # 设置 running variance 为 1。
    bn_dep.running_var.fill_(1.0)

# 固定同一个目标样本。
same_sample = torch.tensor([[1.0, 1.0, 1.0]])
# 构造 batch A：同伴都靠近 0。
peers_a = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [-0.5, -0.5, -0.5]])
# 拼出 batch A。
batch_a = torch.cat([same_sample, peers_a], dim=0)
# 构造 batch B：同伴都靠近 10。
peers_b = torch.tensor([[9.0, 9.0, 9.0], [10.0, 10.0, 10.0], [11.0, 11.0, 11.0]])
# 拼出 batch B。
batch_b = torch.cat([same_sample, peers_b], dim=0)

# 切换训练模式。
bn_dep.train()
# 为了不让第一次前向更新的 running stats 干扰第二次演示，先保存状态。
state_before = {k: v.clone() for k, v in bn_dep.state_dict().items()}
# 同一个样本放在 batch A 中进行训练态归一化。
out_a = bn_dep(batch_a)[0].detach().clone()
# 恢复 BatchNorm 状态，保证 batch B 从同一初始 running state 出发。
bn_dep.load_state_dict(state_before)
# 同一个样本放在 batch B 中进行训练态归一化。
out_b = bn_dep(batch_b)[0].detach().clone()

# 再次恢复固定 running stats。
bn_dep.load_state_dict(state_before)
# 切换评估模式。
bn_dep.eval()
# 评估模式下单独输入同一个样本。
out_eval_1 = bn_dep(same_sample).detach().clone()
# 评估模式下再次输入同一个样本。
out_eval_2 = bn_dep(same_sample).detach().clone()

# 打印训练态 batch A 输出。
print("train output with batch A peers =", out_a)
# 打印训练态 batch B 输出。
print("train output with batch B peers =", out_b)
# 打印两个训练态输出的差异。
print("train same-sample output difference =", float((out_a - out_b).abs().max()))
# 打印评估态第一次输出。
print("eval output #1 =", out_eval_1)
# 打印评估态第二次输出。
print("eval output #2 =", out_eval_2)
# 打印评估态重复输出差异。
print("eval repeat max diff =", float((out_eval_1 - out_eval_2).abs().max()))

# 绘制同一样本在不同 batch 同伴下的训练态输出。
plt.figure(figsize=(7, 4.5))
train_values = [float(out_a[0]), float(out_b[0]), float(out_eval_1[0, 0])]
plt.bar(["train: peers near 0", "train: peers near 10", "eval: running stats"], train_values)
plt.ylabel("Normalized value of feature 0")
plt.title("同一个样本：训练态输出依赖 Batch 同伴，评估态使用 Running Stats")
plt.xticks(rotation=12)
plt.tight_layout()
plt.savefig(FIG_DIR / "lesson46_same_sample_batch_dependence.png", dpi=180)
plt.close()


# -----------------------------------------------------------------------------
# 实验5：Batch 越小，统计量估计越抖
# -----------------------------------------------------------------------------
title("实验5：Batch Size 越小，Batch Mean / Variance 越抖")

# 定义总体分布的真实均值。
true_mean = 3.0
# 定义总体分布的真实标准差。
true_std = 2.0
# 定义需要比较的 Batch Size。
batch_sizes = [2, 4, 8, 16, 32, 64, 128]
# 定义每个 Batch Size 重复抽样多少次。
trials = 2000
# 存储 Batch Mean 的标准差。
mean_estimator_std = []
# 存储 Batch Variance 的标准差。
var_estimator_std = []

# 逐个 Batch Size 做重复抽样实验。
for B in batch_sizes:
    # 一次生成 trials x B 个随机样本。
    samples = np.random.normal(loc=true_mean, scale=true_std, size=(trials, B))
    # 每一行视为一个 batch，计算 batch mean。
    means = samples.mean(axis=1)
    # 每一行视为一个 batch，计算 biased / population batch variance。
    vars_ = samples.var(axis=1, ddof=0)
    # 记录 batch mean 估计量的波动大小。
    mean_estimator_std.append(means.std())
    # 记录 batch variance 估计量的波动大小。
    var_estimator_std.append(vars_.std())
    # 打印数值结果。
    print(
        f"B={B:3d} | std(batch mean)={means.std():.6f} | "
        f"std(batch var)={vars_.std():.6f}"
    )

# 绘制统计量估计波动与 Batch Size 的关系。
plt.figure(figsize=(7, 4.5))
plt.plot(batch_sizes, mean_estimator_std, marker="o", label="std of batch mean estimator")
plt.plot(batch_sizes, var_estimator_std, marker="o", label="std of batch variance estimator")
plt.xscale("log", base=2)
plt.xlabel("Batch size")
plt.ylabel("Estimator variability across repeated batches")
plt.title("Batch 越小，Batch Statistics 的随机波动越大")
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "lesson46_small_batch_statistics_noise.png", dpi=180)
plt.close()


# -----------------------------------------------------------------------------
# 实验6：Batch Size = 1 的边界条件
# -----------------------------------------------------------------------------
title("实验6：BatchNorm1d 的 Batch Size = 1 边界条件")

# 创建一个训练态 BatchNorm1d。
bn_one = nn.BatchNorm1d(3)
# 切换训练模式。
bn_one.train()
# 构造 Shape=(1,3) 的输入，即每个 Channel 只有 1 个值可用于统计。
one_value_per_channel = torch.tensor([[1.0, 2.0, 3.0]])
# 尝试运行，并捕获 PyTorch 给出的错误。
try:
    # 训练态执行 BatchNorm。
    _ = bn_one(one_value_per_channel)
    # 如果没有报错，就打印提示。
    print("Unexpected: BatchNorm1d accepted shape (1,3) in training mode.")
except Exception as exc:
    # 打印异常类型和核心信息。
    print("training with input shape (1,3) raises:", type(exc).__name__)
    # 只打印第一行错误信息，避免输出过长。
    print(str(exc).split("\n")[0])

# BatchNorm1d 也可以接受 (N,C,L)；即便 N=1，只要 N*L > 1，仍有多个值可统计。
sequence_like = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0], [1.0, 3.0, 5.0, 7.0]]])
# 创建新的 BatchNorm1d。
bn_seq = nn.BatchNorm1d(3, affine=False, track_running_stats=False)
# 切换训练模式。
bn_seq.train()
# 执行归一化。
seq_out = bn_seq(sequence_like)
# 打印输入 Shape。
print("input shape (N,C,L) =", tuple(sequence_like.shape))
# 打印输出 Shape。
print("output shape =", tuple(seq_out.shape))
# 打印每个 Channel 在 N 和 L 两个维度合起来后的均值。
print("output channel means =", seq_out.mean(dim=(0, 2)))


# -----------------------------------------------------------------------------
# 实验7：BatchNorm2d 到底沿哪些轴统计
# -----------------------------------------------------------------------------
title("实验7：BatchNorm2d 的统计轴")

# 构造 Shape=(N,C,H,W)=(2,2,2,2) 的四维输入。
X4 = torch.tensor(
    [
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[10.0, 20.0], [30.0, 40.0]],
        ],
        [
            [[5.0, 6.0], [7.0, 8.0]],
            [[50.0, 60.0], [70.0, 80.0]],
        ],
    ]
)
# 创建 BatchNorm2d；关闭 affine 和 running stats，只看当前 batch。
bn2d = nn.BatchNorm2d(2, affine=False, track_running_stats=False)
# 切换训练模式。
bn2d.train()
# 计算 PyTorch 输出。
Y4 = bn2d(X4)
# BatchNorm2d 对每个 Channel 在 N、H、W 三个轴上统计均值。
mean4 = X4.mean(dim=(0, 2, 3))
# BatchNorm2d 对每个 Channel 在 N、H、W 三个轴上统计 biased variance。
var4 = X4.var(dim=(0, 2, 3), unbiased=False)
# 手工归一化；reshape 成 (1,C,1,1) 便于广播。
Y4_manual = (X4 - mean4.view(1, -1, 1, 1)) / torch.sqrt(var4.view(1, -1, 1, 1) + bn2d.eps)
# 计算最大误差。
diff4 = (Y4 - Y4_manual).abs().max().item()

# 打印输入 Shape。
print("X4.shape =", tuple(X4.shape))
# 打印 per-channel mean。
print("per-channel mean over (N,H,W) =", mean4)
# 打印 per-channel variance。
print("per-channel var over (N,H,W)  =", var4)
# 打印手工与 PyTorch 最大误差。
print(f"BatchNorm2d manual vs torch max abs diff = {diff4:.10f}")
# 打印输出在 N,H,W 维度上的均值。
print("output channel means =", Y4.mean(dim=(0, 2, 3)))
# 打印输出在 N,H,W 维度上的总体标准差。
print("output channel stds  =", Y4.std(dim=(0, 2, 3), unbiased=False))

# 绘制 BatchNorm2d 的轴语义示意图。
plt.figure(figsize=(8, 4.5))
plt.axis("off")
plt.text(0.05, 0.80, "Input shape: (N, C, H, W)", fontsize=15)
plt.text(0.05, 0.60, "For each channel C independently:", fontsize=13)
plt.text(0.10, 0.43, "collect values across N, H, W", fontsize=13)
plt.text(0.10, 0.28, "compute mean / variance -> normalize -> gamma / beta", fontsize=13)
plt.text(0.05, 0.10, "Statistics axes = (N, H, W); channel axis C is kept separate.", fontsize=12)
plt.title("BatchNorm2d 的统计轴", fontsize=16)
plt.tight_layout()
plt.savefig(FIG_DIR / "lesson46_batchnorm2d_axes.png", dpi=180)
plt.close()


# -----------------------------------------------------------------------------
# 最终提示
# -----------------------------------------------------------------------------
title("实验完成")
# 打印总结提示。
print("所有实验已完成，PNG 图片已保存到 figures/ 目录。")
