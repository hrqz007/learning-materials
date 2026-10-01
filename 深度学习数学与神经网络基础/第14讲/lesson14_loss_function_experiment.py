# -*- coding: utf-8 -*-
"""
第14讲实验：Loss Function - 模型如何把“错了多少”变成可优化的数字

目标：
1. 区分 Prediction、Target、Error 和 Loss。
2. 观察为什么“带符号误差直接平均”会发生抵消。
3. 区分 per-sample loss、sum loss 和 mean loss。
4. 观察不同 loss 对大误差的惩罚强度不同。
5. 把 Loss 画成参数 w 的函数，为后续 Gradient/Optimizer 做准备。
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)


def section(title):
    print("\n" + "=" * 76)
    print(title)
    print("=" * 76)


def absolute_loss(prediction, target):
    """绝对误差：|prediction - target|。"""
    return np.abs(prediction - target)


def squared_loss(prediction, target):
    """平方误差：(prediction - target)^2。"""
    return (prediction - target) ** 2


# -----------------------------------------------------------------------------
# 实验1：Prediction / Target / Error / Loss 是不同对象
# -----------------------------------------------------------------------------
section("实验1：Prediction、Target、Error 与 Loss")
target = 4.0
predictions = np.array([1.0, 3.0, 4.0, 5.0, 7.0])

for p in predictions:
    error = p - target
    abs_l = absolute_loss(p, target)
    sq_l = squared_loss(p, target)
    print(
        f"prediction={p:>4.1f}  target={target:.1f}  "
        f"error={error:>5.1f}  |error|={abs_l:>4.1f}  squared_loss={sq_l:>5.1f}"
    )

print("\n观察：prediction=4 时 loss 最小；离 target 越远，loss 通常越大。")


# -----------------------------------------------------------------------------
# 实验2：为什么不能直接平均带符号 error
# -----------------------------------------------------------------------------
section("实验2：带符号 Error 会互相抵消")
targets = np.array([4.0, 4.0])
preds = np.array([2.0, 6.0])
errors = preds - targets
print("targets      =", targets)
print("predictions  =", preds)
print("errors       =", errors)
print(f"mean(error)  = {errors.mean():.3f}")
print(f"mean(|error|)= {absolute_loss(preds, targets).mean():.3f}")
print(f"mean(square) = {squared_loss(preds, targets).mean():.3f}")
print("结论：平均带符号误差为 0，并不代表两个预测都正确。")


# -----------------------------------------------------------------------------
# 实验3：per-sample loss、sum loss、mean loss
# -----------------------------------------------------------------------------
section("实验3：单样本 Loss、Sum Loss 与 Mean Loss")
targets = np.array([3.0, 5.0, 2.0, 8.0])
preds = np.array([2.0, 6.0, 2.5, 7.0])
per_sample = squared_loss(preds, targets)
print("targets         =", targets)
print("predictions     =", preds)
print("per-sample loss =", per_sample)
print(f"sum loss         = {per_sample.sum():.6f}")
print(f"mean loss        = {per_sample.mean():.6f}")

# 把同一批数据复制一遍：样本数翻倍。
dup_targets = np.concatenate([targets, targets])
dup_preds = np.concatenate([preds, preds])
dup_loss = squared_loss(dup_preds, dup_targets)
print("\n把完全相同的数据复制一遍（batch size 翻倍）：")
print(f"duplicated sum loss  = {dup_loss.sum():.6f}")
print(f"duplicated mean loss = {dup_loss.mean():.6f}")
print("观察：sum 随样本数翻倍，mean 保持不变。")


# -----------------------------------------------------------------------------
# 实验4：不同 Loss 对大误差的惩罚强度不同
# -----------------------------------------------------------------------------
section("实验4：Absolute Loss 与 Squared Loss 的惩罚强度")
deviations = np.array([0.5, 1.0, 2.0, 4.0])
for d in deviations:
    abs_l = d
    sq_l = d ** 2
    print(f"误差绝对值={d:>3.1f}  absolute_loss={abs_l:>4.1f}  squared_loss={sq_l:>5.1f}")
print("观察：误差从 2 变成 4 时，Absolute Loss 变 2 倍，Squared Loss 变 4 倍。")


# -----------------------------------------------------------------------------
# 实验5：Loss 可以看成参数 w 的函数
# -----------------------------------------------------------------------------
section("实验5：把 Loss 看成参数 w 的函数")
x = 2.0
y_true = 6.0


def model(w):
    # 最简单模型：prediction = w * x
    return w * x


def loss_from_w(w):
    prediction = model(w)
    return squared_loss(prediction, y_true)

for w in [0.0, 1.0, 2.0, 2.5, 3.0, 3.5, 4.0]:
    y_pred = model(w)
    loss = loss_from_w(w)
    print(f"w={w:>3.1f}  prediction={y_pred:>4.1f}  loss={loss:>6.2f}")

print("最小 loss 出现在 w=3，因为此时 prediction=2*3=6，刚好等于 target。")


# -----------------------------------------------------------------------------
# 图1：固定 target 时，prediction 改变会怎样改变 loss
# -----------------------------------------------------------------------------
p_grid = np.linspace(-1, 9, 500)
target_plot = 4.0
abs_curve = absolute_loss(p_grid, target_plot)
sq_curve = squared_loss(p_grid, target_plot)

plt.figure(figsize=(8, 5.5))
plt.plot(p_grid, abs_curve, label="Absolute loss |p-y|")
plt.plot(p_grid, sq_curve, label="Squared loss (p-y)^2")
plt.axvline(target_plot, linestyle="--", linewidth=1, label="Target = 4")
plt.xlabel("Prediction p")
plt.ylabel("Loss")
plt.title("Loss as a Function of Prediction")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
fig1 = FIG_DIR / "lesson14_prediction_loss_curve.png"
plt.savefig(fig1, dpi=180)
plt.close()


# -----------------------------------------------------------------------------
# 图2：固定输入 x 和 target 后，loss 变成参数 w 的函数
# -----------------------------------------------------------------------------
w_grid = np.linspace(-1, 6, 500)
loss_curve = np.array([loss_from_w(w) for w in w_grid])

plt.figure(figsize=(8, 5.5))
plt.plot(w_grid, loss_curve, label="L(w) = (2w - 6)^2")
plt.scatter([3.0], [0.0], zorder=3, label="Minimum: w=3")
plt.xlabel("Weight w")
plt.ylabel("Loss")
plt.title("A Tiny Loss Landscape: Weight -> Loss")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
fig2 = FIG_DIR / "lesson14_weight_loss_landscape.png"
plt.savefig(fig2, dpi=180)
plt.close()

print("\n实验图已保存：")
print(fig1)
print(fig2)
