# -*- coding: utf-8 -*-
"""
第44讲实验：Dropout

实验目标：
1. 观察训练模式下 Dropout 的随机 Mask、零元素比例与 inverted dropout 缩放。
2. 验证 Dropout 在期望意义上保持激活尺度，而 eval 模式下退化为恒等映射。
3. 严格区分 model.train()/model.eval() 与 torch.no_grad()。
4. 在一个小样本、高容量回归任务中比较不同 Dropout 概率的泛化表现。
5. 验证 Dropout 不改变 Tensor Shape，也没有可训练 Parameter。

说明：
- 使用固定随机种子，保证实验可复现。
- 泛化实验中的 Validation 只用于选择 p；Test 仅在最后报告。
"""

from pathlib import Path
import math
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import font_manager

SEED = 20261003
OUT_DIR = Path(__file__).resolve().parent
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

# 中文字体，避免 PNG 乱码。
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_PROP = font_manager.FontProperties(fname=FONT_PATH)
plt.rcParams["font.family"] = FONT_PROP.get_name()
plt.rcParams["axes.unicode_minus"] = False


def set_seed(seed=SEED):
    """固定 Python / NumPy / PyTorch 随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def experiment_inverted_dropout(log):
    """实验1：验证训练模式下的 inverted dropout。"""
    log.append("实验1：随机 Mask、零比例与 inverted dropout 缩放")
    set_seed(1)
    p = 0.5
    dropout = nn.Dropout(p=p)
    x = torch.ones(10000)

    # 训练模式：随机丢弃，并把保留下来的元素放大 1/(1-p)。
    dropout.train()
    y = dropout(x)
    zero_ratio = float((y == 0).float().mean().item())
    nonzero_values = torch.unique(y[y != 0]).tolist()
    mean_y = float(y.mean().item())

    log.append(f"p={p}, 输入均值={x.mean().item():.6f}")
    log.append(f"训练模式零元素比例≈{zero_ratio:.6f}")
    log.append(f"训练模式非零值={nonzero_values}")
    log.append(f"训练模式输出均值≈{mean_y:.6f}")

    # 评估模式：Dropout 关闭，直接返回输入。
    dropout.eval()
    y_eval = dropout(x)
    max_diff_eval = float((y_eval - x).abs().max().item())
    log.append(f"eval 模式 max|Dropout(x)-x|={max_diff_eval:.6g}")
    log.append("")

    # 图：一次训练态输出的前 80 个元素。
    dropout.train()
    set_seed(2)
    y_small = dropout(torch.ones(80)).numpy()
    fig, ax = plt.subplots(figsize=(9.2, 4.5))
    ax.stem(np.arange(len(y_small)), y_small, basefmt=" ")
    ax.set_xlabel("元素索引")
    ax.set_ylabel("Dropout 输出")
    ax.set_title("p=0.5：训练模式下元素随机变为 0 或放大为 2")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "lesson44_dropout_mask.png", dpi=180)
    plt.close(fig)


def experiment_expectation(log):
    """实验2：重复采样，观察 inverted dropout 的期望近似保持不变。"""
    log.append("实验2：为什么训练时要除以 keep probability？")
    set_seed(3)
    p = 0.3
    q = 1 - p
    dropout = nn.Dropout(p=p)
    dropout.train()
    x = torch.tensor([1.0, 2.0, 4.0, 8.0])

    samples = []
    for _ in range(20000):
        samples.append(dropout(x).unsqueeze(0))
    ys = torch.cat(samples, dim=0)
    empirical_mean = ys.mean(dim=0)

    log.append(f"p={p}, keep probability q={q}")
    log.append(f"输入 x={x.tolist()}")
    log.append(f"20000 次 Dropout 后逐元素经验均值={empirical_mean.tolist()}")
    log.append(f"最大期望误差={float((empirical_mean-x).abs().max().item()):.6f}")
    log.append("")

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    idx = np.arange(len(x))
    ax.bar(idx - 0.18, x.numpy(), width=0.36, label="原始输入")
    ax.bar(idx + 0.18, empirical_mean.numpy(), width=0.36, label="训练态多次采样均值")
    ax.set_xticks(idx, ["x1", "x2", "x3", "x4"])
    ax.set_ylabel("数值")
    ax.set_title("Inverted Dropout：训练阶段随机，但期望尺度保持")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "lesson44_expectation.png", dpi=180)
    plt.close(fig)


def experiment_train_eval_no_grad(log):
    """实验3：train/eval 与 no_grad 是两条正交控制轴。"""
    log.append("实验3：model.train()/eval() 与 torch.no_grad() 不是同一件事")
    set_seed(4)
    model = nn.Sequential(nn.Linear(4, 4, bias=False), nn.Dropout(p=0.5))
    with torch.no_grad():
        model[0].weight.copy_(torch.eye(4))
    x = torch.ones(1, 4, requires_grad=True)

    # train + grad：Dropout 开；Autograd 开。
    model.train()
    set_seed(10)
    y_train_grad = model(x)
    train_requires_grad = y_train_grad.requires_grad

    # train + no_grad：Dropout 仍然开；Autograd 关。
    model.train()
    set_seed(10)
    with torch.no_grad():
        y_train_nograd = model(x)
    train_nograd_requires_grad = y_train_nograd.requires_grad

    # eval + grad：Dropout 关；Autograd 仍然开。
    model.eval()
    y_eval_grad = model(x)
    eval_requires_grad = y_eval_grad.requires_grad

    # eval + no_grad：Dropout 关；Autograd 关。
    model.eval()
    with torch.no_grad():
        y_eval_nograd = model(x)

    log.append(f"train + grad 输出={y_train_grad.detach().tolist()}, requires_grad={train_requires_grad}")
    log.append(f"train + no_grad 输出={y_train_nograd.tolist()}, requires_grad={train_nograd_requires_grad}")
    log.append(f"eval + grad 输出={y_eval_grad.detach().tolist()}, requires_grad={eval_requires_grad}")
    log.append(f"eval + no_grad 输出={y_eval_nograd.tolist()}, requires_grad={y_eval_nograd.requires_grad}")
    log.append("同一随机种子下 train+grad 与 train+no_grad 输出是否一致：" + str(torch.allclose(y_train_grad.detach(), y_train_nograd)))
    log.append("")


def experiment_shape_and_parameters(log):
    """实验4：Dropout 不改变 Shape，也没有可训练参数。"""
    log.append("实验4：Shape 与 Parameter")
    set_seed(5)
    dropout = nn.Dropout(p=0.25)
    X = torch.randn(2, 3, 8)
    dropout.train()
    Y = dropout(X)
    param_count = sum(p.numel() for p in dropout.parameters())
    log.append(f"X.shape={tuple(X.shape)}")
    log.append(f"Dropout(X).shape={tuple(Y.shape)}")
    log.append(f"Dropout 可训练参数量={param_count}")
    log.append("")


def make_regression_data():
    """小样本非线性回归：训练集有噪声，Val/Test 使用新的独立样本。"""
    set_seed(100)
    n_train, n_val, n_test = 48, 512, 512

    def f(x):
        return torch.sin(2.5 * x) + 0.25 * x

    X_train = torch.linspace(-2.8, 2.8, n_train).unsqueeze(1)
    y_train = f(X_train) + 0.28 * torch.randn_like(X_train)

    X_val = torch.linspace(-3.0, 3.0, n_val).unsqueeze(1)
    y_val = f(X_val) + 0.28 * torch.randn_like(X_val)

    X_test = torch.linspace(-3.0, 3.0, n_test).unsqueeze(1)
    # 用不同随机种子生成独立测试噪声。
    torch.manual_seed(101)
    y_test = f(X_test) + 0.28 * torch.randn_like(X_test)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), f


class DropoutMLP(nn.Module):
    """高容量 MLP：通过 p 控制 Dropout 强度。"""
    def __init__(self, p):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Dropout(p=p),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=p),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_dropout_model(p, train_data, val_data, steps=3500, lr=0.01):
    """相同初始化与预算，只改变 Dropout p；按 Validation 保存 best checkpoint。"""
    X_train, y_train = train_data
    X_val, y_val = val_data
    set_seed(777)
    model = DropoutMLP(p)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    best_step = -1
    history = []

    for step in range(steps):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        optimizer.step()

        if step % 10 == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad():
                train_eval_loss = float(loss_fn(model(X_train), y_train).item())
                val_loss = float(loss_fn(model(X_val), y_val).item())
            history.append((step + 1, train_eval_loss, val_loss))
            if val_loss < best_val:
                best_val = val_loss
                best_step = step + 1
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    return model, best_val, best_step, history


def experiment_generalization(log):
    """实验5：Dropout 强度与泛化的关系。"""
    log.append("实验5：Dropout 强度与 Generalization（只按 Validation 选 p）")
    train_data, val_data, test_data, f = make_regression_data()
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data
    loss_fn = nn.MSELoss()

    p_values = [0.0, 0.1, 0.3, 0.6]
    rows = []
    models = {}
    histories = {}

    for p in p_values:
        model, best_val, best_step, history = train_dropout_model(p, train_data, val_data)
        with torch.no_grad():
            train_mse = float(loss_fn(model(X_train), y_train).item())
            val_mse = float(loss_fn(model(X_val), y_val).item())
            test_mse = float(loss_fn(model(X_test), y_test).item())
        rows.append((p, train_mse, val_mse, test_mse, best_step))
        models[p] = model
        histories[p] = history

    best = min(rows, key=lambda r: r[2])
    log.append("p | train MSE | val MSE | test MSE | best step")
    for row in rows:
        log.append(f"{row[0]:.1f} | {row[1]:.6f} | {row[2]:.6f} | {row[3]:.6f} | {row[4]}")
    log.append(f"按 Validation 选择的最佳 p={best[0]:.1f}, 独立 Test MSE={best[3]:.6f}")
    log.append("")

    # 图1：不同 p 的 Train / Val / Test MSE。
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    idx = np.arange(len(rows))
    ax.plot(idx, [r[1] for r in rows], marker="o", label="Train MSE")
    ax.plot(idx, [r[2] for r in rows], marker="o", label="Validation MSE")
    ax.plot(idx, [r[3] for r in rows], marker="o", label="Test MSE（教学展示）")
    ax.set_xticks(idx, [str(p) for p in p_values])
    ax.set_xlabel("Dropout probability p")
    ax.set_ylabel("MSE")
    ax.set_title("Dropout 太弱可能过拟合；太强可能欠拟合")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "lesson44_dropout_generalization.png", dpi=180)
    plt.close(fig)

    # 图2：p=0 与最佳 p 的函数拟合对比。
    grid = torch.linspace(-3.2, 3.2, 500).unsqueeze(1)
    with torch.no_grad():
        true_curve = f(grid).squeeze(1).numpy()
        pred0 = models[0.0](grid).squeeze(1).numpy()
        predb = models[best[0]](grid).squeeze(1).numpy()
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    ax.scatter(X_train.squeeze(1).numpy(), y_train.squeeze(1).numpy(), s=20, alpha=0.65, label="Train samples")
    ax.plot(grid.squeeze(1).numpy(), true_curve, linewidth=2, label="Underlying function")
    ax.plot(grid.squeeze(1).numpy(), pred0, linewidth=1.8, label="p=0.0")
    ax.plot(grid.squeeze(1).numpy(), predb, linewidth=1.8, label=f"best p={best[0]:.1f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Dropout 对小样本高容量 MLP 拟合形状的影响")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "lesson44_fit_comparison.png", dpi=180)
    plt.close(fig)

    # 图3：p=0 与最佳 p 的训练/验证曲线。
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    for p in [0.0, best[0]]:
        h = histories[p]
        steps = [x[0] for x in h]
        train_losses = [x[1] for x in h]
        val_losses = [x[2] for x in h]
        ax.plot(steps, train_losses, label=f"p={p:.1f} Train")
        ax.plot(steps, val_losses, linestyle="--", label=f"p={p:.1f} Validation")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("MSE (eval mode)")
    ax.set_yscale("log")
    ax.set_title("Dropout 下应在 eval 模式计算 Validation")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "lesson44_train_val_curves.png", dpi=180)
    plt.close(fig)


def main():
    set_seed()
    log = []
    log.append("=" * 82)
    log.append("第44讲实验：Dropout")
    log.append("=" * 82)
    log.append(f"PyTorch version: {torch.__version__}")
    log.append("")

    experiment_inverted_dropout(log)
    experiment_expectation(log)
    experiment_train_eval_no_grad(log)
    experiment_shape_and_parameters(log)
    experiment_generalization(log)

    output_path = OUT_DIR / "第44讲_实验实际输出.txt"
    output_path.write_text("\n".join(log), encoding="utf-8")
    print("\n".join(log))


if __name__ == "__main__":
    main()
