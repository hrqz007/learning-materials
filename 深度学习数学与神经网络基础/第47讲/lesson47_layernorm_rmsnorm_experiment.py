# -*- coding: utf-8 -*-
"""
第47讲实验：LayerNorm 与 RMSNorm
面向 LLM / Agent Systems 科研的深度学习零基础课

实验目标：
1. 手算并验证 LayerNorm。
2. 手算并验证 RMSNorm。
3. 观察 LayerNorm 的“去均值”与 RMSNorm 的“只按 RMS 缩放”的差异。
4. 验证 LayerNorm / RMSNorm 对 Batch 同伴不敏感，而 BatchNorm 在训练态会受 Batch 同伴影响。
5. 验证 LayerNorm / RMSNorm 的 train/eval 数学行为一致（没有 running statistics）。
6. 在 LLM 常见的 (B,T,D) Tensor 上验证：LayerNorm 通常沿最后一个 Hidden Dimension D 统计。
7. 检查可学习参数 Shape，并给出 Transformer Pre-Norm / Post-Norm 的最小代码示意。
"""

from pathlib import Path
import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import font_manager

# -----------------------------
# 0. 固定随机种子，保证实验可复现
# -----------------------------
SEED = 47
np.random.seed(SEED)
torch.manual_seed(SEED)

# -----------------------------
# 1. 输出目录
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# 2. Matplotlib 中文字体
# -----------------------------
font_candidates = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
]
for font_path in font_candidates:
    if Path(font_path).exists():
        font_manager.fontManager.addfont(font_path)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False


def print_section(title: str) -> None:
    """打印分隔标题，便于阅读实验输出。"""
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def manual_layer_norm(x: torch.Tensor, eps: float = 1e-5):
    """
    手工实现 LayerNorm（只沿最后一个维度）。

    输入：
        x: (..., D)
    输出：
        y: (..., D)
        mean: (..., 1)
        var: (..., 1)
    """
    mean = x.mean(dim=-1, keepdim=True)
    # LayerNorm 使用总体方差形式，即 unbiased=False。
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    y = (x - mean) / torch.sqrt(var + eps)
    return y, mean, var


def manual_rms_norm(x: torch.Tensor, eps: float = 1e-5):
    """
    手工实现 RMSNorm（只沿最后一个维度）。

    RMS(x) = sqrt(mean(x^2) + eps)
    y = x / RMS(x)
    """
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    y = x / rms
    return y, rms


# ============================================================================
# 实验 1：手算 LayerNorm，并与 PyTorch 对照
# ============================================================================
print_section("实验1：手算 LayerNorm，并与 PyTorch nn.LayerNorm 对照")

x = torch.tensor([1.0, 2.0, 3.0, 4.0])
eps = 1e-5

ln_manual, mean_manual, var_manual = manual_layer_norm(x, eps=eps)
ln = nn.LayerNorm(4, elementwise_affine=False, eps=eps)
ln_torch = ln(x)

print(f"输入 x = {x.tolist()}")
print(f"mean = {mean_manual.item():.6f}")
print(f"variance(unbiased=False) = {var_manual.item():.6f}")
print(f"std = {math.sqrt(var_manual.item() + eps):.6f}")
print(f"手工 LayerNorm = {ln_manual.tolist()}")
print(f"PyTorch LayerNorm = {ln_torch.tolist()}")
print(f"最大绝对误差 = {(ln_manual - ln_torch).abs().max().item():.10f}")
print(f"LayerNorm 后均值 = {ln_torch.mean().item():.8f}")
print(f"LayerNorm 后方差 = {ln_torch.var(unbiased=False).item():.8f}")

# ============================================================================
# 实验 2：手算 RMSNorm，并与 PyTorch 对照
# ============================================================================
print_section("实验2：手算 RMSNorm，并与 PyTorch nn.RMSNorm 对照")

rms_manual, rms_value = manual_rms_norm(x, eps=eps)
rms_module = nn.RMSNorm(4, eps=eps, elementwise_affine=False)
rms_torch = rms_module(x)

print(f"输入 x = {x.tolist()}")
print(f"RMS = {rms_value.item():.6f}")
print(f"手工 RMSNorm = {rms_manual.tolist()}")
print(f"PyTorch RMSNorm = {rms_torch.tolist()}")
print(f"最大绝对误差 = {(rms_manual - rms_torch).abs().max().item():.10f}")
print(f"RMSNorm 后均值 = {rms_torch.mean().item():.8f}")
print(f"RMSNorm 后 RMS = {torch.sqrt(torch.mean(rms_torch ** 2)).item():.8f}")

# ============================================================================
# 实验 3：LayerNorm 与 RMSNorm 对“整体平移”的反应不同
# ============================================================================
print_section("实验3：整体加常数——LayerNorm 与 RMSNorm 的关键差异")

x_shift = x + 100.0
ln_x = nn.LayerNorm(4, elementwise_affine=False, eps=eps)(x)
ln_shift = nn.LayerNorm(4, elementwise_affine=False, eps=eps)(x_shift)
rms_x = nn.RMSNorm(4, elementwise_affine=False, eps=eps)(x)
rms_shift = nn.RMSNorm(4, elementwise_affine=False, eps=eps)(x_shift)

print(f"原始 x = {x.tolist()}")
print(f"平移后 x+100 = {x_shift.tolist()}")
print(f"LayerNorm(x) = {ln_x.tolist()}")
print(f"LayerNorm(x+100) = {ln_shift.tolist()}")
print(f"LayerNorm 最大差异 = {(ln_x - ln_shift).abs().max().item():.10f}")
print(f"RMSNorm(x) = {rms_x.tolist()}")
print(f"RMSNorm(x+100) = {rms_shift.tolist()}")
print(f"RMSNorm 最大差异 = {(rms_x - rms_shift).abs().max().item():.10f}")

# 图1：原始、LayerNorm、RMSNorm 对比
fig = plt.figure(figsize=(8, 4.5))
positions = np.arange(4)
width = 0.25
plt.bar(positions - width, x.numpy(), width=width, label="原始 x")
plt.bar(positions, ln_x.detach().numpy(), width=width, label="LayerNorm")
plt.bar(positions + width, rms_x.detach().numpy(), width=width, label="RMSNorm")
plt.xticks(positions, ["d1", "d2", "d3", "d4"])
plt.xlabel("Hidden Dimension")
plt.ylabel("数值")
plt.title("LayerNorm 与 RMSNorm：同一 Token 内部的数值变换")
plt.legend()
plt.tight_layout()
fig.savefig(FIG_DIR / "lesson47_layernorm_rmsnorm_values.png", dpi=180)
plt.close(fig)

# 图2：平移不变性对比
fig = plt.figure(figsize=(8, 4.5))
ln_diff = (ln_x - ln_shift).abs().detach().numpy()
rms_diff = (rms_x - rms_shift).abs().detach().numpy()
positions = np.arange(4)
plt.bar(positions - 0.18, ln_diff, width=0.36, label="LayerNorm |差异|")
plt.bar(positions + 0.18, rms_diff, width=0.36, label="RMSNorm |差异|")
plt.xticks(positions, ["d1", "d2", "d3", "d4"])
plt.ylabel("|Norm(x) - Norm(x+100)|")
plt.title("整体平移 +100 后：LayerNorm 基本不变，RMSNorm 会变化")
plt.legend()
plt.tight_layout()
fig.savefig(FIG_DIR / "lesson47_shift_invariance.png", dpi=180)
plt.close(fig)

# ============================================================================
# 实验 4：同一个样本换 Batch 同伴——BatchNorm 会变，LN/RMSNorm 不变
# ============================================================================
print_section("实验4：换 Batch 同伴——BatchNorm 与 LayerNorm/RMSNorm 的本质差异")

target = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
companions_a = torch.tensor([
    [0.0, 0.0, 0.0, 0.0],
    [2.0, 2.0, 2.0, 2.0],
    [3.0, 3.0, 3.0, 3.0],
])
companions_b = torch.tensor([
    [100.0, 100.0, 100.0, 100.0],
    [120.0, 120.0, 120.0, 120.0],
    [150.0, 150.0, 150.0, 150.0],
])

batch_a = torch.cat([target, companions_a], dim=0)
batch_b = torch.cat([target, companions_b], dim=0)

# BatchNorm：每个 feature/channel 沿 Batch 统计。
bn_a = nn.BatchNorm1d(4, affine=False, track_running_stats=False)
bn_b = nn.BatchNorm1d(4, affine=False, track_running_stats=False)
bn_a.train()
bn_b.train()
bn_target_a = bn_a(batch_a)[0]
bn_target_b = bn_b(batch_b)[0]

# LayerNorm / RMSNorm：每个样本自己沿最后一维统计，与 Batch 同伴无关。
ln_target_a = nn.LayerNorm(4, elementwise_affine=False)(batch_a)[0]
ln_target_b = nn.LayerNorm(4, elementwise_affine=False)(batch_b)[0]
rms_target_a = nn.RMSNorm(4, elementwise_affine=False)(batch_a)[0]
rms_target_b = nn.RMSNorm(4, elementwise_affine=False)(batch_b)[0]

print(f"同一 target = {target[0].tolist()}")
print(f"BatchNorm target in batch A = {bn_target_a.tolist()}")
print(f"BatchNorm target in batch B = {bn_target_b.tolist()}")
print(f"BatchNorm 最大差异 = {(bn_target_a - bn_target_b).abs().max().item():.6f}")
print(f"LayerNorm 最大差异 = {(ln_target_a - ln_target_b).abs().max().item():.10f}")
print(f"RMSNorm 最大差异 = {(rms_target_a - rms_target_b).abs().max().item():.10f}")

# 图3：同样本换 Batch 同伴后的输出差异
fig = plt.figure(figsize=(8, 4.5))
methods = ["BatchNorm", "LayerNorm", "RMSNorm"]
diffs = [
    (bn_target_a - bn_target_b).abs().max().item(),
    (ln_target_a - ln_target_b).abs().max().item(),
    (rms_target_a - rms_target_b).abs().max().item(),
]
plt.bar(methods, diffs)
plt.ylabel("同一样本换 Batch 后的最大输出差异")
plt.title("Batch Dependency：BatchNorm vs LayerNorm / RMSNorm")
plt.tight_layout()
fig.savefig(FIG_DIR / "lesson47_batch_dependency.png", dpi=180)
plt.close(fig)

# ============================================================================
# 实验 5：train/eval——LayerNorm 与 RMSNorm 不依赖 running statistics
# ============================================================================
print_section("实验5：train() / eval() 对 LayerNorm 与 RMSNorm 的影响")

sample_3d = torch.randn(2, 3, 4)
ln_affine = nn.LayerNorm(4)
rms_affine = nn.RMSNorm(4)

ln_affine.train()
ln_train = ln_affine(sample_3d)
ln_affine.eval()
ln_eval = ln_affine(sample_3d)

rms_affine.train()
rms_train = rms_affine(sample_3d)
rms_affine.eval()
rms_eval = rms_affine(sample_3d)

print(f"LayerNorm train/eval 最大差异 = {(ln_train - ln_eval).abs().max().item():.10f}")
print(f"RMSNorm train/eval 最大差异 = {(rms_train - rms_eval).abs().max().item():.10f}")
print("说明：在没有 Dropout 等其他随机模块时，LN/RMSNorm 自身的数学行为不因 train/eval 改变。")

# ============================================================================
# 实验 6：(B,T,D)——逐 Token 沿 Hidden Dimension D 归一化
# ============================================================================
print_section("实验6：LLM 常见 (B,T,D) Tensor 上的 LayerNorm / RMSNorm")

B, T, D = 2, 3, 4
hidden = torch.tensor([
    [[1.0, 2.0, 3.0, 4.0],
     [2.0, 4.0, 6.0, 8.0],
     [-1.0, 0.0, 1.0, 2.0]],
    [[10.0, 20.0, 30.0, 40.0],
     [3.0, 3.0, 6.0, 12.0],
     [-4.0, -2.0, 0.0, 2.0]],
])

ln3 = nn.LayerNorm(D, elementwise_affine=False, eps=eps)
rms3 = nn.RMSNorm(D, elementwise_affine=False, eps=eps)
ln_hidden = ln3(hidden)
rms_hidden = rms3(hidden)

ln_means = ln_hidden.mean(dim=-1)
ln_vars = ln_hidden.var(dim=-1, unbiased=False)
rms_values = torch.sqrt(torch.mean(rms_hidden ** 2, dim=-1))
rms_means = rms_hidden.mean(dim=-1)

print(f"hidden.shape = {tuple(hidden.shape)}")
print(f"LayerNorm output shape = {tuple(ln_hidden.shape)}")
print(f"RMSNorm output shape = {tuple(rms_hidden.shape)}")
print("LayerNorm 每个 (B,T) 位置沿 D 的 mean：")
print(ln_means)
print("LayerNorm 每个 (B,T) 位置沿 D 的 variance：")
print(ln_vars)
print("RMSNorm 每个 (B,T) 位置沿 D 的 RMS：")
print(rms_values)
print("RMSNorm 每个 (B,T) 位置沿 D 的 mean（通常不要求为0）：")
print(rms_means)

# 图4：每个 Token 的 LayerNorm mean / var 与 RMSNorm rms
fig = plt.figure(figsize=(9, 4.8))
positions = np.arange(B * T)
labels = [f"b{b}t{t}" for b in range(B) for t in range(T)]
plt.plot(positions, ln_means.reshape(-1).numpy(), marker="o", label="LayerNorm mean")
plt.plot(positions, ln_vars.reshape(-1).numpy(), marker="o", label="LayerNorm variance")
plt.plot(positions, rms_values.reshape(-1).numpy(), marker="o", label="RMSNorm RMS")
plt.xticks(positions, labels)
plt.ylabel("统计量")
plt.title("(B,T,D) 中：每个 Token 独立沿 D 统计")
plt.legend()
plt.tight_layout()
fig.savefig(FIG_DIR / "lesson47_tokenwise_stats.png", dpi=180)
plt.close(fig)

# ============================================================================
# 实验 7：可学习参数 Shape
# ============================================================================
print_section("实验7：LayerNorm / RMSNorm 的可学习参数 Shape")

ln_param = nn.LayerNorm(D, elementwise_affine=True)
rms_param = nn.RMSNorm(D, elementwise_affine=True)

print(f"LayerNorm.weight.shape = {tuple(ln_param.weight.shape)}")
print(f"LayerNorm.bias.shape = {tuple(ln_param.bias.shape)}")
print(f"RMSNorm.weight.shape = {tuple(rms_param.weight.shape)}")
print(f"RMSNorm 是否有 bias 属性 = {hasattr(rms_param, 'bias')}")
print(f"LayerNorm 参数量 = {sum(p.numel() for p in ln_param.parameters())}")
print(f"RMSNorm 参数量 = {sum(p.numel() for p in rms_param.parameters())}")

# ============================================================================
# 实验 8：最小 Pre-Norm / Post-Norm 结构示意
# ============================================================================
print_section("实验8：Transformer 中 Pre-Norm / Post-Norm 的最小代码示意")

class TinyResidualBlock(nn.Module):
    """一个最小残差块，仅用于展示 Norm 放置位置，不代表真实 Transformer。"""

    def __init__(self, d_model: int, pre_norm: bool):
        super().__init__()
        self.pre_norm = pre_norm
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm:
            # Pre-Norm: x + F(Norm(x))
            return x + self.ff(self.norm(x))
        # Post-Norm: Norm(x + F(x))
        return self.norm(x + self.ff(x))

pre = TinyResidualBlock(D, pre_norm=True)
post = TinyResidualBlock(D, pre_norm=False)

# 为了只比较结构，把两者线性层和 Norm 参数对齐。
with torch.no_grad():
    post.ff.weight.copy_(pre.ff.weight)
    post.norm.weight.copy_(pre.norm.weight)
    post.norm.bias.copy_(pre.norm.bias)

pre_out = pre(hidden)
post_out = post(hidden)

print(f"Pre-Norm input/output shape: {tuple(hidden.shape)} -> {tuple(pre_out.shape)}")
print(f"Post-Norm input/output shape: {tuple(hidden.shape)} -> {tuple(post_out.shape)}")
print("注意：本实验只展示 Norm 的放置位置，不据此宣称哪一种结构在所有任务上都更好。")

# 图5：BN/LN/RMSNorm 的统计轴概念图（以 B,T,D 为例）
fig = plt.figure(figsize=(9, 5.2))
ax = plt.gca()
ax.axis("off")
ax.text(0.05, 0.88, "假设 Hidden States 的 Shape = (B, T, D)", fontsize=13, weight="bold")
ax.text(0.08, 0.68, "BatchNorm（概念类比）", fontsize=12)
ax.text(0.38, 0.68, "同一 feature/channel 跨样本（及空间/位置）统计", fontsize=11)
ax.text(0.08, 0.48, "LayerNorm", fontsize=12)
ax.text(0.38, 0.48, "每个 Token 自己沿最后的 Hidden Dimension D 统计 mean + variance", fontsize=11)
ax.text(0.08, 0.28, "RMSNorm", fontsize=12)
ax.text(0.38, 0.28, "每个 Token 自己沿 D 统计 RMS；不先减均值", fontsize=11)
ax.text(0.08, 0.09, "关键问题：Normalization 到底沿哪些 Axis 计算统计量？", fontsize=12, weight="bold")
plt.tight_layout()
fig.savefig(FIG_DIR / "lesson47_normalization_axes.png", dpi=180)
plt.close(fig)

print_section("实验总结")
print("1. LayerNorm：每个样本/Token 独立沿最后 D 维计算 mean 与 variance。")
print("2. RMSNorm：每个样本/Token 独立沿最后 D 维计算 RMS，不做 mean-centering。")
print("3. LayerNorm 对统一加常数近似不变；RMSNorm 一般不是。")
print("4. LN/RMSNorm 不依赖 Batch 同伴，也没有 BatchNorm 那样的 running statistics。")
print("5. 在 (B,T,D) 的 Transformer hidden states 上，常见用法是 normalized_shape=D。")
print("6. LN 通常有 gamma 与 beta；PyTorch RMSNorm 默认只有可学习 weight(gamma)。")
print("7. Pre-Norm / Post-Norm 的区别是 Norm 与 Residual/SubLayer 的相对位置，不是两种不同的归一化公式。")
