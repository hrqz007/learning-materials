# 第33讲实验：LoRA 的低秩直觉
# 目标：比较完整参数更新 ΔW 与低秩 B@A 近似。
# 这不是训练真实 LLM，只用小矩阵展示 LoRA 为什么省参数。

import numpy as np

np.random.seed(33)

d_in = 16
d_out = 16

# Frozen base weight
W0 = np.random.randn(d_out, d_in) * 0.2

# Construct a "true" task delta that is itself rank-2 + a little noise.
B_true = np.random.randn(d_out, 2)
A_true = np.random.randn(2, d_in)
delta_true = B_true @ A_true + 0.03 * np.random.randn(d_out, d_in)

def best_rank_r_approx(M, r):
    # SVD gives the best rank-r approximation under Frobenius norm.
    U, S, VT = np.linalg.svd(M, full_matrices=False)
    return (U[:, :r] * S[:r]) @ VT[:r, :]

full_params = d_out * d_in

print("Toy LoRA low-rank approximation demo")
print("=" * 72)
print(f"Base matrix shape: {d_out} x {d_in}")
print(f"Full trainable parameters: {full_params}")
print()

full_error = np.mean((delta_true - delta_true) ** 2)
print(f"Full fine-tuning delta error: {full_error:.8f}")

for r in [1, 2, 4, 8]:
    approx = best_rank_r_approx(delta_true, r)
    mse = np.mean((delta_true - approx) ** 2)
    lora_params = r * d_in + d_out * r
    ratio = lora_params / full_params
    print(
        f"rank={r:2d} | LoRA params={lora_params:3d} "
        f"| ratio={ratio*100:6.2f}% | delta MSE={mse:.6f}"
    )

print()
print("Lesson:")
print("- Full fine-tuning can represent any delta but trains all parameters.")
print("- LoRA restricts the delta to a low-rank subspace.")
print("- If the useful task update is approximately low-rank, small r can work well.")
