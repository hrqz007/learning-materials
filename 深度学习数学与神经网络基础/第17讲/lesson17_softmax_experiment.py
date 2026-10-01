# -*- coding: utf-8 -*-
"""
第17讲实验：Softmax - 怎样把 Logits 变成概率分布

运行：
    python lesson17_softmax_experiment.py

依赖：
    numpy
    matplotlib
    torch（可选；若未安装，会自动跳过 PyTorch 对照）
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# 固定打印格式，便于观察小数。
np.set_printoptions(precision=6, suppress=True)

# 所有实验图都保存到脚本旁边的 figures 文件夹。
OUTPUT_DIR = Path(__file__).resolve().parent / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def softmax_naive(logits):
    """直接按公式实现 Softmax；大 logits 时可能数值溢出。"""
    exp_values = np.exp(logits)
    return exp_values / np.sum(exp_values, axis=-1, keepdims=True)


def softmax_stable(logits):
    """数值稳定版 Softmax：每组 logits 先减去该组最大值。"""
    logits = np.asarray(logits, dtype=np.float64)
    max_value = np.max(logits, axis=-1, keepdims=True)
    shifted = logits - max_value
    exp_values = np.exp(shifted)
    probabilities = exp_values / np.sum(exp_values, axis=-1, keepdims=True)
    return probabilities


def entropy(probabilities):
    """计算自然对数版本的熵，只用于比较分布尖锐程度。"""
    p = np.asarray(probabilities, dtype=np.float64)
    return -np.sum(p * np.log(p + 1e-12), axis=-1)


print("=" * 72)
print("第17讲：Softmax 实验")
print("=" * 72)

# ----------------------------------------------------------------------
# 实验1：手工拆开 Softmax 的每一步
# ----------------------------------------------------------------------
print("\n[实验1] 手工拆开 Softmax")
logits = np.array([2.0, 1.0, 0.1], dtype=np.float64)
exp_values = np.exp(logits)
exp_sum = np.sum(exp_values)
probs = exp_values / exp_sum
print("logits                 =", logits)
print("exp(logits)            =", exp_values)
print("sum(exp(logits))       =", exp_sum)
print("probabilities          =", probs)
print("sum(probabilities)     =", np.sum(probs))
print("argmax(logits)         =", np.argmax(logits))
print("argmax(probabilities)  =", np.argmax(probs))

# ----------------------------------------------------------------------
# 实验2：Softmax 对“整体平移”不敏感
# ----------------------------------------------------------------------
print("\n[实验2] Softmax 的平移不变性")
for c in [10.0, -7.5, 100.0]:
    shifted_logits = logits + c
    shifted_probs = softmax_stable(shifted_logits)
    diff = np.max(np.abs(shifted_probs - softmax_stable(logits)))
    print(f"加常数 c={c:6.1f} -> probs={shifted_probs}, max_diff={diff:.12f}")

# ----------------------------------------------------------------------
# 实验3：直接 exp 大数会溢出；稳定版不会
# ----------------------------------------------------------------------
print("\n[实验3] 数值稳定性：为什么要先减最大值")
big_logits = np.array([1000.0, 1001.0, 999.0])
with np.errstate(over="ignore", invalid="ignore"):
    naive_exp = np.exp(big_logits)
    naive_probs = naive_exp / np.sum(naive_exp)
stable_probs = softmax_stable(big_logits)
print("big logits             =", big_logits)
print("naive exp              =", naive_exp)
print("naive probabilities    =", naive_probs)
print("stable probabilities   =", stable_probs)
print("stable sum             =", np.sum(stable_probs))

# ----------------------------------------------------------------------
# 实验4：Temperature 改变分布尖锐程度
# ----------------------------------------------------------------------
print("\n[实验4] Temperature")
temperatures = [0.5, 1.0, 2.0]
for T in temperatures:
    p = softmax_stable(logits / T)
    print(
        f"T={T:<3} -> probs={p}, max_prob={np.max(p):.6f}, entropy={float(entropy(p)):.6f}"
    )

# ----------------------------------------------------------------------
# 实验5：Batch / 行方向 Softmax
# ----------------------------------------------------------------------
print("\n[实验5] Batch Softmax：沿最后一个维度归一化")
batch_logits = np.array(
    [
        [2.0, 1.0, 0.1],
        [0.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)
batch_probs = softmax_stable(batch_logits)
print("batch logits shape     =", batch_logits.shape)
print("batch probabilities    =\n", batch_probs)
print("row sums               =", np.sum(batch_probs, axis=-1))

# ----------------------------------------------------------------------
# 实验6：模拟 LLM 的 vocabulary logits
# ----------------------------------------------------------------------
print("\n[实验6] 微型 LLM vocabulary Softmax")
tokens = np.array(["猫", "狗", "AI", "你好", "世界"])
vocab_logits = np.array([1.2, 2.1, -0.3, 0.8, 1.0], dtype=np.float64)
vocab_probs = softmax_stable(vocab_logits)
for token, logit, p in zip(tokens, vocab_logits, vocab_probs):
    print(f"token={token:<2}  logit={logit:>5.2f}  probability={p:.6f}")
best_index = int(np.argmax(vocab_probs))
print("最高概率 token          =", tokens[best_index])
print("最高概率                =", vocab_probs[best_index])

# ----------------------------------------------------------------------
# 实验7：NumPy 与 PyTorch 对照
# ----------------------------------------------------------------------
print("\n[实验7] NumPy 与 PyTorch 对照")
try:
    import torch

    torch_logits = torch.tensor([2.0, 1.0, 0.1], dtype=torch.float64)
    torch_probs = torch.softmax(torch_logits, dim=-1).detach().cpu().numpy()
    numpy_probs = softmax_stable(logits)
    print("NumPy probs             =", numpy_probs)
    print("PyTorch probs           =", torch_probs)
    print("max abs difference      =", np.max(np.abs(numpy_probs - torch_probs)))
except Exception as exc:
    print("未完成 PyTorch 对照：", repr(exc))

# ----------------------------------------------------------------------
# 图1：logits 与 Softmax 概率
# ----------------------------------------------------------------------
fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111)
indices = np.arange(len(logits))
ax.bar(indices, probs)
ax.set_xticks(indices)
ax.set_xticklabels(["class 0\nlogit=2.0", "class 1\nlogit=1.0", "class 2\nlogit=0.1"])
ax.set_ylim(0, 1)
ax.set_ylabel("Softmax probability")
ax.set_title("Softmax turns logits into a probability distribution")
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "lesson17_logits_to_probabilities.png", dpi=180)
plt.close(fig)

# ----------------------------------------------------------------------
# 图2：Temperature 对概率分布的影响
# ----------------------------------------------------------------------
fig = plt.figure(figsize=(9, 5))
ax = fig.add_subplot(111)
width = 0.22
for offset, T in zip([-width, 0.0, width], temperatures):
    p = softmax_stable(logits / T)
    ax.bar(indices + offset, p, width=width, label=f"T={T}")
ax.set_xticks(indices)
ax.set_xticklabels(["class 0", "class 1", "class 2"])
ax.set_ylim(0, 1)
ax.set_ylabel("Probability")
ax.set_title("Temperature changes sharpness, not class order")
ax.legend()
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "lesson17_temperature_effect.png", dpi=180)
plt.close(fig)

# ----------------------------------------------------------------------
# 图3：二分类时，logit 差距怎样变成概率差距
# 对 logits=[d, 0]，第一类概率就是 1/(1+exp(-d))。
# ----------------------------------------------------------------------
gaps = np.linspace(-8, 8, 400)
first_class_probs = 1.0 / (1.0 + np.exp(-gaps))
fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111)
ax.plot(gaps, first_class_probs)
ax.axhline(0.5, linewidth=1)
ax.axvline(0.0, linewidth=1)
ax.set_xlabel("Logit gap: z1 - z2")
ax.set_ylabel("Probability of class 1")
ax.set_title("In two classes, Softmax depends on the logit gap")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "lesson17_logit_gap_probability.png", dpi=180)
plt.close(fig)

print("\n图已保存到：", OUTPUT_DIR)
print("- lesson17_logits_to_probabilities.png")
print("- lesson17_temperature_effect.png")
print("- lesson17_logit_gap_probability.png")
print("\n实验结束。")
