# -*- coding: utf-8 -*-
"""
第42讲实验：Train / Validation / Test 与 Data Leakage

实验目标：
1. 演示正确的 Train -> Validation -> Test 工作流。
2. 演示反复在 Validation 上选模型会产生 selection bias / validation overfitting。
3. 演示 row-level split 在“同一实体/同一 Agent trajectory 有多条相似记录”时会产生严重泄漏。

依赖：numpy, matplotlib
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

SEED = 42
rng = np.random.default_rng(SEED)
OUT_DIR = Path(__file__).resolve().parent
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

# 设置中文字体，避免保存 PNG 时中文乱码。
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_PROP = font_manager.FontProperties(fname=FONT_PATH)
plt.rcParams["font.family"] = FONT_PROP.get_name()
plt.rcParams["axes.unicode_minus"] = False


def mse(y_true, y_pred):
    """均方误差。"""
    return float(np.mean((y_true - y_pred) ** 2))


def polynomial_features(x, degree):
    """把一维 x 变成 [1, x, x^2, ..., x^degree]。"""
    return np.column_stack([x ** p for p in range(degree + 1)])


def fit_polynomial(x, y, degree):
    """最小二乘拟合多项式，不依赖 sklearn。"""
    X = polynomial_features(x, degree)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def predict_polynomial(x, coef):
    """使用拟合好的多项式系数预测。"""
    degree = len(coef) - 1
    return polynomial_features(x, degree) @ coef


def experiment_1_correct_split_and_model_selection(log):
    """实验1：Train 训练，Validation 选 degree，Test 最终一次评估。"""
    log.append("=" * 72)
    log.append("实验1：正确的 Train -> Validation -> Test 工作流")
    log.append("=" * 72)

    n = 600
    x = rng.uniform(-1.0, 1.0, size=n)
    noise = rng.normal(0.0, 0.18, size=n)
    y = np.sin(2 * np.pi * x) + noise

    # 固定随机划分：60% / 20% / 20%。
    idx = rng.permutation(n)
    n_train = int(0.60 * n)
    n_val = int(0.20 * n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    degrees = [1, 3, 5, 7, 9, 12, 15]
    rows = []
    for d in degrees:
        coef = fit_polynomial(x_train, y_train, d)
        train_loss = mse(y_train, predict_polynomial(x_train, coef))
        val_loss = mse(y_val, predict_polynomial(x_val, coef))
        # 为了教学展示可以计算所有 test loss，但“选模型”绝不使用它。
        test_loss = mse(y_test, predict_polynomial(x_test, coef))
        rows.append((d, train_loss, val_loss, test_loss, coef))

    best = min(rows, key=lambda r: r[2])  # 只按 Validation MSE 选。
    best_degree, best_train, best_val, best_test, best_coef = best

    log.append(f"Dataset size = {n}")
    log.append(f"Train / Val / Test = {len(train_idx)} / {len(val_idx)} / {len(test_idx)}")
    log.append("候选 degree 的 MSE：")
    for d, tr, va, te, _ in rows:
        log.append(f"  degree={d:2d} | train={tr:.6f} | val={va:.6f} | test(仅教学记录)={te:.6f}")
    log.append(f"按 Validation 选择的最佳 degree = {best_degree}")
    log.append(f"最终一次 Test MSE = {best_test:.6f}")

    # 绘图：训练/验证/测试误差 vs degree。
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(degrees, [r[1] for r in rows], marker="o", label="Train MSE")
    ax.plot(degrees, [r[2] for r in rows], marker="o", label="Validation MSE")
    ax.plot(degrees, [r[3] for r in rows], marker="o", label="Test MSE（仅教学展示）")
    ax.set_xlabel("Polynomial Degree")
    ax.set_ylabel("MSE")
    ax.set_title("Train / Validation / Test 的不同职责")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "lesson42_model_selection_mse.png", dpi=180)
    plt.close(fig)

    # 绘图：最佳模型在 test 区域上的拟合。
    x_grid = np.linspace(-1, 1, 400)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x_train, y_train, s=12, alpha=0.35, label="Train")
    ax.scatter(x_val, y_val, s=18, alpha=0.65, label="Validation")
    ax.scatter(x_test, y_test, s=18, alpha=0.65, label="Test")
    ax.plot(x_grid, np.sin(2 * np.pi * x_grid), linewidth=2, label="真实函数")
    ax.plot(x_grid, predict_polynomial(x_grid, best_coef), linewidth=2, label=f"Val 选出的 degree={best_degree}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("只用 Validation 选择模型，Test 留到最后")
    ax.legend(ncol=2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "lesson42_selected_model_fit.png", dpi=180)
    plt.close(fig)

    return {
        "best_degree": best_degree,
        "best_val": best_val,
        "best_test": best_test,
        "rows": rows,
    }


def experiment_2_validation_overfitting(log):
    """实验2：即使没有真实能力，候选模型越多，Validation 最优值也会越来越乐观。"""
    log.append("")
    log.append("=" * 72)
    log.append("实验2：反复在 Validation 上挑最优，也会发生 Validation Overfitting")
    log.append("=" * 72)

    # 完全随机二分类：不存在可学习规律。
    n_val = 200
    n_test = 5000
    y_val = rng.integers(0, 2, size=n_val)
    y_test = rng.integers(0, 2, size=n_test)

    candidate_counts = [1, 10, 100, 1000, 5000]
    repeats = 80
    avg_best_val = []
    avg_selected_test = []
    std_best_val = []
    std_selected_test = []

    for m in candidate_counts:
        best_vals = []
        selected_tests = []
        for _ in range(repeats):
            # 每个候选模型都只是随机猜测，没有任何真实预测能力。
            val_preds = rng.integers(0, 2, size=(m, n_val), dtype=np.int8)
            val_accs = (val_preds == y_val).mean(axis=1)
            best_idx = int(np.argmax(val_accs))

            # 只对被选中的那个“模型”生成独立 test 预测。
            test_pred = rng.integers(0, 2, size=n_test, dtype=np.int8)
            test_acc = float((test_pred == y_test).mean())

            best_vals.append(float(val_accs[best_idx]))
            selected_tests.append(test_acc)

        avg_best_val.append(float(np.mean(best_vals)))
        avg_selected_test.append(float(np.mean(selected_tests)))
        std_best_val.append(float(np.std(best_vals)))
        std_selected_test.append(float(np.std(selected_tests)))

    for m, va, te in zip(candidate_counts, avg_best_val, avg_selected_test):
        log.append(f"候选模型数={m:5d} | 平均最佳 Validation Acc={va:.4f} | 该模型平均 Test Acc={te:.4f}")

    # 绘图：候选数越多，Validation 最优值越“漂亮”，Test 仍在 0.5。
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(candidate_counts, avg_best_val, yerr=std_best_val, marker="o", capsize=4, label="Best Validation Accuracy")
    ax.errorbar(candidate_counts, avg_selected_test, yerr=std_selected_test, marker="o", capsize=4, label="Selected Model Test Accuracy")
    ax.axhline(0.5, linestyle="--", linewidth=1.5, label="Chance = 0.5")
    ax.set_xscale("log")
    ax.set_xlabel("尝试的候选模型 / 超参数数量")
    ax.set_ylabel("Accuracy")
    ax.set_title("Validation 也会被反复调参“过拟合”")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "lesson42_validation_overfitting.png", dpi=180)
    plt.close(fig)

    return {
        "candidate_counts": candidate_counts,
        "avg_best_val": avg_best_val,
        "avg_selected_test": avg_selected_test,
    }


def nearest_neighbor_predict(x_train, y_train, x_query):
    """最简单的 1-NN：寻找欧氏距离最近的训练样本。"""
    # x_query: [Q, D], x_train: [N, D]
    d2 = ((x_query[:, None, :] - x_train[None, :, :]) ** 2).sum(axis=2)
    nn_idx = np.argmin(d2, axis=1)
    return y_train[nn_idx]


def experiment_3_group_leakage(log):
    """实验3：row split vs group/trajectory split。"""
    log.append("")
    log.append("=" * 72)
    log.append("实验3：同一实体 / 同一 Agent trajectory 跨 split 导致严重泄漏")
    log.append("=" * 72)

    n_groups = 200
    records_per_group = 6
    dim = 10

    # 每个 group 有一个随机中心与一个随机标签；标签与中心之间没有可泛化规律。
    group_centers = rng.normal(0, 1, size=(n_groups, dim))
    group_labels = rng.integers(0, 2, size=n_groups)

    xs, ys, group_ids = [], [], []
    for gid in range(n_groups):
        for _ in range(records_per_group):
            # 同一 group 内记录高度相似，模拟同一用户/病人/trajectory 的多个 timestep/窗口。
            xs.append(group_centers[gid] + rng.normal(0, 0.025, size=dim))
            ys.append(group_labels[gid])
            group_ids.append(gid)

    X = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.int64)
    g = np.asarray(group_ids, dtype=np.int64)
    n = len(y)

    # A. 错误做法：按 row 随机切分，同一个 group 很可能同时出现在 train/test。
    perm = rng.permutation(n)
    cut = int(0.7 * n)
    train_idx = perm[:cut]
    test_idx = perm[cut:]
    pred_row = nearest_neighbor_predict(X[train_idx], y[train_idx], X[test_idx])
    row_acc = float((pred_row == y[test_idx]).mean())
    train_groups = set(g[train_idx].tolist())
    test_groups = set(g[test_idx].tolist())
    overlap = len(train_groups & test_groups)

    # B. 正确做法：先按 group 切分，再把整个 group 放入某一个 split。
    group_perm = rng.permutation(n_groups)
    group_cut = int(0.7 * n_groups)
    train_group_set = set(group_perm[:group_cut].tolist())
    test_group_set = set(group_perm[group_cut:].tolist())
    train_mask = np.array([gid in train_group_set for gid in g])
    test_mask = np.array([gid in test_group_set for gid in g])
    pred_group = nearest_neighbor_predict(X[train_mask], y[train_mask], X[test_mask])
    group_acc = float((pred_group == y[test_mask]).mean())

    log.append(f"总 group 数 = {n_groups}, 每组记录数 = {records_per_group}, 总记录数 = {n}")
    log.append(f"错误 row-level split: train/test 共享 group 数 = {overlap} / {len(test_groups)} test groups")
    log.append(f"错误 row-level split 的 Test Accuracy = {row_acc:.4f}")
    log.append(f"正确 group-level split 的 Test Accuracy = {group_acc:.4f}")
    log.append("由于 group label 是随机生成的，真正面对 unseen groups 时理论上只能接近 chance=0.5。")

    # 绘图：错误划分 vs 正确划分。
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = ["Row-level split\n(泄漏)", "Group/Trajectory split\n(正确)"]
    values = [row_acc, group_acc]
    bars = ax.bar(labels, values)
    ax.axhline(0.5, linestyle="--", linewidth=1.5, label="Chance = 0.5")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Test Accuracy")
    ax.set_title("同一实体/Trajectory 跨 Split 会制造虚假的高分")
    ax.legend()
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.025, f"{val:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "lesson42_group_leakage.png", dpi=180)
    plt.close(fig)

    return {"row_acc": row_acc, "group_acc": group_acc, "overlap": overlap}


def main():
    log = []
    log.append("第42讲实验实际输出")
    log.append(f"Random seed = {SEED}")
    log.append("")

    r1 = experiment_1_correct_split_and_model_selection(log)
    r2 = experiment_2_validation_overfitting(log)
    r3 = experiment_3_group_leakage(log)

    log.append("")
    log.append("=" * 72)
    log.append("实验结论摘要")
    log.append("=" * 72)
    log.append(f"1) Validation 选出的 degree = {r1['best_degree']}，最终 Test MSE = {r1['best_test']:.6f}")
    log.append(f"2) 随候选数增加，最佳 Validation Acc 可被随机选择抬高，但 Test 仍接近 0.5。")
    log.append(f"3) Row-level leakage accuracy = {r3['row_acc']:.4f}，Group-level split accuracy = {r3['group_acc']:.4f}")
    log.append("4) 结论：Split 不是简单的随机切行；必须按真实独立单位、时间因果和评估边界设计。")

    text = "\n".join(log) + "\n"
    output_path = OUT_DIR / "第42讲_实验实际输出.txt"
    output_path.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
