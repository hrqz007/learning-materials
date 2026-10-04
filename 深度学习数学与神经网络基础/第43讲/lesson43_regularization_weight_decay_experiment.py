# -*- coding: utf-8 -*-
"""
第43讲实验：Regularization 与 Weight Decay

实验目标：
1. 在高维、样本较少的回归任务中观察“训练拟合很好但泛化较差”。
2. 扫描不同 weight_decay，观察 Training / Validation / Test MSE 与参数范数的变化。
3. 验证适度正则化可能改善泛化，但过强正则化会导致欠拟合。
4. 观察 Weight Decay 对“真正有用特征”和“噪声特征”权重的影响。

说明：
- 为了把“正则化”本身讲清楚，本实验使用最基础的 SGD + weight_decay。
- 对 vanilla SGD，weight_decay 与 L2 风格的参数收缩关系最直观。
- 第39讲已经从 Optimizer 机制解释了 AdamW；本讲重点转向 Generalization。
"""

from pathlib import Path
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import font_manager

SEED_DATA = 123
SEED_MODEL = 999
D = 100
N_TRAIN = 120
N_VAL = 1000
N_TEST = 1000
NOISE_STD = 0.5
LR = 0.02
STEPS = 8000
WEIGHT_DECAYS = [0.0, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]

OUT_DIR = Path(__file__).resolve().parent
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

# 中文字体，避免 PNG 乱码。
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_PROP = font_manager.FontProperties(fname=FONT_PATH)
plt.rcParams["font.family"] = FONT_PROP.get_name()
plt.rcParams["axes.unicode_minus"] = False


def make_dataset():
    """生成高维回归数据：100 个特征中只有前 8 个真正有用。"""
    torch.manual_seed(SEED_DATA)
    np.random.seed(SEED_DATA)

    w_true = torch.zeros(D, 1)
    w_true[:8] = torch.tensor(
        [[2.0], [-1.5], [1.0], [0.5], [-0.8], [1.2], [-0.4], [0.7]],
        dtype=torch.float32,
    )

    def gen(n):
        X = torch.randn(n, D)
        y = X @ w_true + NOISE_STD * torch.randn(n, 1)
        return X, y

    return w_true, gen(N_TRAIN), gen(N_VAL), gen(N_TEST)


def train_one(weight_decay, X_train, y_train):
    """同一初始化、同一 LR、同一步数，只改变 weight_decay。"""
    torch.manual_seed(SEED_MODEL)
    model = torch.nn.Linear(D, 1, bias=False)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LR,
        weight_decay=weight_decay,
    )
    loss_fn = torch.nn.MSELoss()

    losses = []
    for step in range(STEPS):
        optimizer.zero_grad()
        pred = model(X_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        optimizer.step()
        if step in {0, 9, 99, 999, STEPS - 1}:
            losses.append((step + 1, float(loss.item())))

    return model, losses


def mse(model, X, y):
    with torch.no_grad():
        return float(torch.nn.functional.mse_loss(model(X), y).item())


def parameter_l2_norm(model):
    total = 0.0
    with torch.no_grad():
        for p in model.parameters():
            total += float((p ** 2).sum().item())
    return math.sqrt(total)


def main():
    w_true, train_data, val_data, test_data = make_dataset()
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data

    log = []
    log.append("=" * 78)
    log.append("第43讲实验：Regularization 与 Weight Decay")
    log.append("=" * 78)
    log.append(f"Train / Val / Test = {N_TRAIN} / {N_VAL} / {N_TEST}")
    log.append(f"Feature dimension D = {D}, 真正有用特征数 = 8, 噪声特征数 = {D-8}")
    log.append(f"Noise std = {NOISE_STD}, LR = {LR}, Steps = {STEPS}")
    log.append("")

    rows = []
    models = {}
    checkpoints = {}

    for wd in WEIGHT_DECAYS:
        model, trace = train_one(wd, X_train, y_train)
        tr = mse(model, X_train, y_train)
        va = mse(model, X_val, y_val)
        te = mse(model, X_test, y_test)
        norm = parameter_l2_norm(model)
        w = model.weight.detach().view(-1).clone()
        informative_abs = float(w[:8].abs().mean().item())
        nuisance_abs = float(w[8:].abs().mean().item())
        cosine_true = float(torch.nn.functional.cosine_similarity(w, w_true.view(-1), dim=0).item())
        rows.append({
            "wd": wd,
            "train": tr,
            "val": va,
            "test": te,
            "norm": norm,
            "informative_abs": informative_abs,
            "nuisance_abs": nuisance_abs,
            "cosine_true": cosine_true,
        })
        models[wd] = model
        checkpoints[wd] = trace

    best = min(rows, key=lambda r: r["val"])

    log.append("实验1：Weight Decay Sweep（只按 Validation 选择）")
    log.append("weight_decay | train MSE | val MSE | test MSE | ||w||2 | mean|w_info| | mean|w_noise| | cos(w, w_true)")
    for r in rows:
        log.append(
            f"{r['wd']:>11g} | {r['train']:.6f} | {r['val']:.6f} | {r['test']:.6f} | "
            f"{r['norm']:.6f} | {r['informative_abs']:.6f} | {r['nuisance_abs']:.6f} | {r['cosine_true']:.6f}"
        )
    log.append("")
    log.append(f"按 Validation MSE 选择的最佳 weight_decay = {best['wd']:g}")
    log.append(f"对应 Validation MSE = {best['val']:.6f}")
    log.append(f"对应独立 Test MSE = {best['test']:.6f}")
    log.append("")

    # 输出几个关键配置的训练轨迹。
    log.append("实验2：同一训练预算下的部分 Training Loss 轨迹")
    for wd in [0.0, 0.1, 1.0]:
        log.append(f"weight_decay={wd:g}")
        for step, loss in checkpoints[wd]:
            log.append(f"  step={step:4d} | train_loss={loss:.6f}")
    log.append("")

    # 实验3：显式计算 L2 penalty 的直觉。
    demo_w = torch.tensor([3.0, 4.0])
    data_loss = 2.0
    lam = 0.1
    l2_penalty = 0.5 * lam * float((demo_w ** 2).sum().item())
    total_objective = data_loss + l2_penalty
    log.append("实验3：L2 Regularization Objective 的数字例子")
    log.append(f"w = [3, 4], ||w||^2 = 25")
    log.append(f"data_loss = {data_loss:.4f}, lambda = {lam:.4f}")
    log.append(f"L2 penalty = (lambda/2) * ||w||^2 = {l2_penalty:.4f}")
    log.append(f"total objective = data_loss + penalty = {total_objective:.4f}")
    log.append("")

    # 图1：Train/Val/Test MSE vs wd。0 用离散索引画，避免 log(0)。
    labels = ["0", "1e-4", "1e-3", "1e-2", "3e-2", "1e-1", "3e-1", "1"]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax.plot(x, [r["train"] for r in rows], marker="o", label="Train MSE")
    ax.plot(x, [r["val"] for r in rows], marker="o", label="Validation MSE")
    ax.plot(x, [r["test"] for r in rows], marker="o", label="Test MSE（仅教学展示）")
    ax.set_xticks(x, labels)
    ax.set_xlabel("Weight Decay")
    ax.set_ylabel("MSE")
    ax.set_title("适度 Weight Decay 可能改善泛化；过强则导致欠拟合")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "lesson43_weight_decay_sweep.png", dpi=180)
    plt.close(fig)

    # 图2：参数范数随 wd 变化。
    fig, ax = plt.subplots(figsize=(8.2, 5.1))
    ax.plot(x, [r["norm"] for r in rows], marker="o")
    ax.set_xticks(x, labels)
    ax.set_xlabel("Weight Decay")
    ax.set_ylabel("Parameter L2 Norm")
    ax.set_title("Weight Decay 增大时，参数整体尺度趋向减小")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "lesson43_parameter_norm.png", dpi=180)
    plt.close(fig)

    # 图3：真实有用特征与噪声特征的平均权重绝对值。
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(x, [r["informative_abs"] for r in rows], marker="o", label="真正有用特征：平均 |w|")
    ax.plot(x, [r["nuisance_abs"] for r in rows], marker="o", label="噪声特征：平均 |w|")
    ax.set_xticks(x, labels)
    ax.set_xlabel("Weight Decay")
    ax.set_ylabel("Mean Absolute Weight")
    ax.set_title("Regularization 会收缩参数，但过强时有用信号也会被压缩")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "lesson43_informative_vs_noise_weights.png", dpi=180)
    plt.close(fig)

    # 图4：选择三个典型 wd 的 learned weights。
    selected = [0.0, 0.1, 1.0]
    fig, axes = plt.subplots(3, 1, figsize=(9.2, 8.2), sharex=True)
    for ax, wd in zip(axes, selected):
        learned = models[wd].weight.detach().view(-1).numpy()
        ax.bar(np.arange(D), np.abs(learned))
        ax.axvline(7.5, linestyle="--", linewidth=1.2)
        ax.set_ylabel("|w|")
        ax.set_title(f"weight_decay={wd:g}")
        ax.grid(True, axis="y", alpha=0.25)
    axes[-1].set_xlabel("Feature Index（0-7 为真实有用特征；8-99 为噪声特征）")
    fig.suptitle("不同 Weight Decay 下的参数分布", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(FIG_DIR / "lesson43_weight_profiles.png", dpi=180)
    plt.close(fig)

    # 图5：Regularization 概念图：data loss + penalty。
    w_grid = np.linspace(-5, 5, 400)
    data_curve = (w_grid - 3.0) ** 2
    lam_demo = 0.4
    penalty_curve = 0.5 * lam_demo * w_grid ** 2
    total_curve = data_curve + penalty_curve
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.plot(w_grid, data_curve, label="Data Loss: (w-3)^2")
    ax.plot(w_grid, penalty_curve, label="Regularization Penalty")
    ax.plot(w_grid, total_curve, label="Total Objective")
    ax.set_xlabel("w")
    ax.set_ylabel("Value")
    ax.set_title("Regularization 改变的是训练目标，而不仅是训练后再检查参数大小")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "lesson43_regularized_objective.png", dpi=180)
    plt.close(fig)

    output_path = OUT_DIR / "第43讲_实验实际输出.txt"
    output_path.write_text("\n".join(log), encoding="utf-8")
    print("\n".join(log))
    print(f"\n实验输出已保存到: {output_path}")
    print(f"实验图已保存到: {FIG_DIR}")


if __name__ == "__main__":
    main()
