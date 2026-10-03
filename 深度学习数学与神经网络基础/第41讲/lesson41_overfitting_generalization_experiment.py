# 第41讲实验：Overfitting 与 Generalization
# 运行环境：Python 3 + NumPy + PyTorch + Matplotlib
# 目标：用最小可复现实验观察“训练集拟合越来越好，但未见数据表现反而变差”的过拟合现象。

from pathlib import Path  # 用于构造跨平台文件路径
import copy               # 用于保存最佳验证集 checkpoint 的模型状态
import numpy as np        # 用于数值处理
import torch              # PyTorch 主库
import torch.nn as nn     # 神经网络层与 Loss
import matplotlib.pyplot as plt  # 绘图
from matplotlib import font_manager  # 设置中文字体

# 固定随机种子，保证实验可复现。
torch.manual_seed(41)
np.random.seed(41)
# 限制线程数，使小实验在不同环境下更稳定、更轻量。
torch.set_num_threads(1)

# 输出目录就是当前脚本所在目录。
OUT_DIR = Path(__file__).resolve().parent
# 单独建立 figures 文件夹保存图片。
FIG_DIR = OUT_DIR / "figures"
# 如果 figures 不存在，就自动创建。
FIG_DIR.mkdir(exist_ok=True)

# 注册支持中文的 Noto Sans CJK 字体，避免图片中文字变成方框。
font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(font_path)
# 读取字体名称。
_cjk_name = font_manager.FontProperties(fname=font_path).get_name()
# Matplotlib 全局使用该中文字体。
plt.rcParams["font.family"] = _cjk_name
# 允许坐标轴正常显示负号。
plt.rcParams["axes.unicode_minus"] = False


def section(title):
    """打印清晰的实验分隔线。"""
    print("\n" + "=" * 82)
    print(title)
    print("=" * 82)


def true_function(x):
    """定义我们希望模型学习的真实规律。"""
    return torch.sin(3.0 * x) + 0.3 * x


def make_regression_data(n, noise_std, seed):
    """生成同一分布下的回归数据：x 均匀采样，y = 真函数 + 高斯噪声。"""
    # 为每个数据集单独使用固定 Generator，避免训练顺序改变后数据也跟着改变。
    g = torch.Generator().manual_seed(seed)
    # 在 [-1, 1] 均匀生成 n 个输入。
    x = torch.rand(n, 1, generator=g) * 2.0 - 1.0
    # 计算没有噪声时的真实函数值。
    y_true = true_function(x)
    # 给真实函数加独立高斯噪声，模拟真实数据中的测量误差或不可解释噪声。
    y = y_true + noise_std * torch.randn(n, 1, generator=g)
    # 同时返回带噪声标签和真实函数，后面绘图时两者都要使用。
    return x, y, y_true


def build_model(kind, seed=123):
    """构造三种容量不同的模型。"""
    # 固定初始化种子，使每次运行相同模型都有相同起点。
    torch.manual_seed(seed)
    # 线性模型容量最低，只能学一条直线。
    if kind == "linear":
        return nn.Linear(1, 1)
    # 小 MLP 能表达非线性，但容量有限。
    if kind == "small":
        return nn.Sequential(
            nn.Linear(1, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
        )
    # 大 MLP 参数更多，更容易把小训练集里的噪声也记住。
    if kind == "large":
        return nn.Sequential(
            nn.Linear(1, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
    # 如果传入未知名称，主动报错，避免静默得到错误实验。
    raise ValueError(f"未知模型类型: {kind}")


def train_model(kind, x_train, y_train, x_val, y_val, epochs=2000, lr=0.01):
    """训练一个模型，并完整记录 train/validation loss 与最佳验证集 checkpoint。"""
    # 创建模型。
    model = build_model(kind)
    # 使用 Adam，确保本讲重点放在泛化而不是优化器差异上。
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # 回归任务使用 MSE。
    loss_fn = nn.MSELoss()
    # 保存每个 Epoch 的训练损失。
    train_losses = []
    # 保存每个 Epoch 的验证损失。
    val_losses = []
    # 初始化最佳验证损失为正无穷。
    best_val = float("inf")
    # 记录最佳验证损失出现在哪个 Epoch。
    best_epoch = -1
    # 保存最佳 Epoch 对应的模型参数。
    best_state = None

    # 开始按 Epoch 训练。
    for epoch in range(epochs):
        # 切到训练模式。
        model.train()
        # 清空上一步梯度。
        optimizer.zero_grad()
        # Forward：在训练集上得到预测。
        pred = model(x_train)
        # 计算训练 MSE。
        train_loss = loss_fn(pred, y_train)
        # Backward：计算所有参数梯度。
        train_loss.backward()
        # 真正修改参数。
        optimizer.step()

        # 切到评估模式。
        model.eval()
        # 验证不需要 Gradient，关闭 Autograd 节省资源。
        with torch.no_grad():
            # 用当前参数在验证集做预测。
            val_pred = model(x_val)
            # 计算验证 MSE。
            val_loss = loss_fn(val_pred, y_val)

        # 保存本 Epoch 的训练损失。
        train_losses.append(float(train_loss.detach()))
        # 保存本 Epoch 的验证损失。
        val_losses.append(float(val_loss))

        # 如果验证损失刷新历史最低值，就保存 checkpoint。
        if float(val_loss) < best_val:
            # 更新最佳验证损失。
            best_val = float(val_loss)
            # 更新最佳 Epoch。
            best_epoch = epoch
            # 深拷贝 state_dict，防止后续训练修改同一块 Tensor。
            best_state = copy.deepcopy(model.state_dict())

    # 返回最终模型、曲线、最佳 Epoch 与最佳参数。
    return model, np.array(train_losses), np.array(val_losses), best_epoch, best_val, best_state


# -----------------------------------------------------------------------------
# 实验1：同一数据上比较欠拟合、较合适容量和高容量模型
# -----------------------------------------------------------------------------
section("实验1：模型容量与 Underfitting / Overfitting")

# 小训练集：只有 24 个样本，并且标签带噪声。
x_train, y_train, y_train_true = make_regression_data(24, noise_std=0.30, seed=101)
# 验证集独立采样，来自同一数据分布。
x_val, y_val, _ = make_regression_data(200, noise_std=0.30, seed=102)
# 测试集同样独立采样，只在最后报告，不参与挑 Epoch。
x_test, y_test, _ = make_regression_data(500, noise_std=0.30, seed=103)
# MSE Loss 对象用于最后统一评估。
loss_fn = nn.MSELoss()

# 保存三种模型的结果，后面用于表格与绘图。
capacity_results = {}

# 依次训练线性、小 MLP、大 MLP。
for kind in ["linear", "small", "large"]:
    # 训练模型并保存全过程。
    model, tr_curve, va_curve, best_epoch, best_val, best_state = train_model(
        kind, x_train, y_train, x_val, y_val, epochs=2000, lr=0.01
    )
    # 最终模型进入评估模式。
    model.eval()
    # 计算训练、验证、测试最终 Loss。
    with torch.no_grad():
        final_train = float(loss_fn(model(x_train), y_train))
        final_val = float(loss_fn(model(x_val), y_val))
        final_test = float(loss_fn(model(x_test), y_test))
    # 建立同结构的最佳验证 checkpoint 模型。
    best_model = build_model(kind)
    # 加载最佳验证 Epoch 的参数。
    best_model.load_state_dict(best_state)
    # 切换为评估模式。
    best_model.eval()
    # 计算最佳 checkpoint 在测试集上的 Loss。
    with torch.no_grad():
        best_test = float(loss_fn(best_model(x_test), y_test))
    # 保存所有指标。
    capacity_results[kind] = {
        "model": model,
        "best_model": best_model,
        "train_curve": tr_curve,
        "val_curve": va_curve,
        "best_epoch": best_epoch,
        "best_val": best_val,
        "final_train": final_train,
        "final_val": final_val,
        "final_test": final_test,
        "best_test": best_test,
    }
    # 打印结果，便于直接阅读 TXT。
    print(
        f"{kind:>6s} | final train={final_train:.6f} | final val={final_val:.6f} "
        f"| best val={best_val:.6f} @ epoch={best_epoch} | "
        f"best test={best_test:.6f} | final test={final_test:.6f}"
    )

# 建立密集网格，用来观察不同模型学到的函数形状。
x_grid = torch.linspace(-1.0, 1.0, 500).reshape(-1, 1)
# 计算真实无噪声函数。
y_grid_true = true_function(x_grid)

# 绘制容量对拟合形状的影响。
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), sharey=True)
# 三个子图分别对应三种容量。
for ax, kind, title in zip(
    axes,
    ["linear", "small", "large"],
    ["低容量：容易欠拟合", "中等容量：较平滑", "高容量：更容易追噪声"],
):
    # 读取最终模型。
    model = capacity_results[kind]["model"]
    # 不记录 Gradient，只画预测曲线。
    with torch.no_grad():
        y_grid_pred = model(x_grid)
    # 画真实规律。
    ax.plot(x_grid.numpy(), y_grid_true.numpy(), label="真实规律")
    # 画训练样本散点。
    ax.scatter(x_train.numpy(), y_train.numpy(), s=28, alpha=0.75, label="训练样本")
    # 画模型最终预测。
    ax.plot(x_grid.numpy(), y_grid_pred.numpy(), label="最终模型")
    # 设置标题。
    ax.set_title(title)
    # 设置横轴。
    ax.set_xlabel("x")
    # 打开轻微网格，帮助读曲线。
    ax.grid(alpha=0.2)
# 只在第一张图设置纵轴。
axes[0].set_ylabel("y")
# 只显示一次图例，减少拥挤。
axes[0].legend(fontsize=9)
# 自动调整布局。
fig.tight_layout()
# 保存图片。
fig.savefig(FIG_DIR / "lesson41_capacity_fits.png", dpi=180)
# 关闭 Figure，释放内存。
plt.close(fig)


# -----------------------------------------------------------------------------
# 实验2：高容量模型的 Train Loss 继续下降，但 Validation Loss 先降后升
# -----------------------------------------------------------------------------
section("实验2：Train / Validation 曲线与 Generalization Gap")

# 取大模型结果。
large = capacity_results["large"]
# 找到最佳验证 Epoch。
best_epoch = int(large["best_epoch"])
# 计算最终 generalization gap：同一个 MSE 下 Val - Train。
final_gap = large["final_val"] - large["final_train"]
# 计算最佳验证 Epoch 当时的训练 Loss。
best_train_loss = float(large["train_curve"][best_epoch])
# 计算最佳 Epoch 的 gap。
best_gap = large["best_val"] - best_train_loss

# 打印关键诊断数字。
print(f"Large model best validation epoch = {best_epoch}")
print(f"Best epoch train loss = {best_train_loss:.6f}")
print(f"Best epoch val loss   = {large['best_val']:.6f}")
print(f"Best epoch gap        = {best_gap:.6f}")
print(f"Final train loss      = {large['final_train']:.6f}")
print(f"Final val loss        = {large['final_val']:.6f}")
print(f"Final gap             = {final_gap:.6f}")
print(f"Best checkpoint test  = {large['best_test']:.6f}")
print(f"Final checkpoint test = {large['final_test']:.6f}")

# 画 Train / Validation Loss 曲线。
fig, ax = plt.subplots(figsize=(8.5, 5.2))
# 为了避免极小训练 Loss 挤在底部，使用对数纵轴。
ax.plot(large["train_curve"], label="Train Loss")
# 画验证损失。
ax.plot(large["val_curve"], label="Validation Loss")
# 标出最佳验证 Epoch。
ax.axvline(best_epoch, linestyle="--", linewidth=1.5, label=f"Best Val Epoch={best_epoch}")
# 使用 log scale 展示数量级变化。
ax.set_yscale("log")
# 横轴为 Epoch。
ax.set_xlabel("Epoch")
# 纵轴为 MSE。
ax.set_ylabel("MSE (log scale)")
# 图标题。
ax.set_title("高容量模型：Train Loss 持续下降，但 Validation Loss 先降后升")
# 显示图例。
ax.legend()
# 轻微网格。
ax.grid(alpha=0.2)
# 自动布局。
fig.tight_layout()
# 保存图片。
fig.savefig(FIG_DIR / "lesson41_train_val_curves.png", dpi=180)
# 关闭图片。
plt.close(fig)

# 画 Generalization Gap 随训练变化。
fig, ax = plt.subplots(figsize=(8.5, 5.0))
# 逐 Epoch 计算 Val - Train。
gap_curve = large["val_curve"] - large["train_curve"]
# 画 gap 曲线。
ax.plot(gap_curve)
# 标出最佳验证 Epoch。
ax.axvline(best_epoch, linestyle="--", linewidth=1.5, label="最佳验证 Epoch")
# 画 0 参考线。
ax.axhline(0.0, linewidth=1.0)
# 横轴名称。
ax.set_xlabel("Epoch")
# 纵轴名称。
ax.set_ylabel("Validation Loss - Train Loss")
# 标题。
ax.set_title("Generalization Gap 会随着过拟合逐渐扩大")
# 图例。
ax.legend()
# 网格。
ax.grid(alpha=0.2)
# 布局。
fig.tight_layout()
# 保存。
fig.savefig(FIG_DIR / "lesson41_generalization_gap.png", dpi=180)
# 关闭。
plt.close(fig)


# -----------------------------------------------------------------------------
# 实验3：Early Stopping 的基本机制——用 Validation 选 checkpoint，而不是用 Test 调参
# -----------------------------------------------------------------------------
section("实验3：Early Stopping / Best Validation Checkpoint")

# 读取大模型最终状态。
final_model = large["model"]
# 读取大模型最佳验证状态。
best_model = large["best_model"]
# 在网格上比较最终模型和最佳验证模型的函数。
with torch.no_grad():
    final_curve = final_model(x_grid)
    best_curve = best_model(x_grid)

# 打印测试集差异。
print(f"用 Validation 选择的 checkpoint: test MSE = {large['best_test']:.6f}")
print(f"训练到最后的 checkpoint:       test MSE = {large['final_test']:.6f}")
print("注意：Test 只用于最后报告，不参与挑 best epoch。")

# 绘制 best checkpoint 与 final checkpoint。
fig, ax = plt.subplots(figsize=(8.5, 5.2))
# 真实无噪声规律。
ax.plot(x_grid.numpy(), y_grid_true.numpy(), label="真实规律")
# 训练样本。
ax.scatter(x_train.numpy(), y_train.numpy(), s=30, alpha=0.72, label="训练样本")
# 最佳验证 checkpoint。
ax.plot(x_grid.numpy(), best_curve.numpy(), label=f"Best Val Checkpoint (epoch {best_epoch})")
# 最终 checkpoint。
ax.plot(x_grid.numpy(), final_curve.numpy(), linestyle="--", label="Final Checkpoint")
# 横轴。
ax.set_xlabel("x")
# 纵轴。
ax.set_ylabel("y")
# 标题。
ax.set_title("Early Stopping：最佳验证模型通常比过度训练后的模型更平滑")
# 图例。
ax.legend(fontsize=9)
# 网格。
ax.grid(alpha=0.2)
# 布局。
fig.tight_layout()
# 保存。
fig.savefig(FIG_DIR / "lesson41_early_stopping_fit.png", dpi=180)
# 关闭。
plt.close(fig)


# -----------------------------------------------------------------------------
# 实验4：Random Labels——高容量网络可以“记住”没有规律的训练标签
# -----------------------------------------------------------------------------
section("实验4：Random Labels Memorization")

# 固定随机种子生成随机输入和随机二分类标签。
torch.manual_seed(410)
# 训练集只有 64 个随机二维点。
x_rand_train = torch.randn(64, 2)
# 标签与输入没有真实关系，纯随机 0/1。
y_rand_train = torch.randint(0, 2, (64,))
# 验证集是独立随机点。
x_rand_val = torch.randn(512, 2)
# 验证标签同样独立随机。
y_rand_val = torch.randint(0, 2, (512,))

# 再固定模型初始化种子。
torch.manual_seed(411)
# 构造足够高容量的分类 MLP。
random_model = nn.Sequential(
    nn.Linear(2, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 2),
)
# 使用 Adam 优化随机标签。
random_optimizer = torch.optim.Adam(random_model.parameters(), lr=0.01)
# 二分类使用 Cross Entropy。
ce = nn.CrossEntropyLoss()
# 记录训练准确率。
train_acc_curve = []
# 记录验证准确率。
val_acc_curve = []
# 记录训练损失。
random_loss_curve = []

# 训练 1000 Epoch，足够展示记忆能力。
for epoch in range(1000):
    # 训练模式。
    random_model.train()
    # 清梯度。
    random_optimizer.zero_grad()
    # Forward 得到 logits。
    logits = random_model(x_rand_train)
    # 计算随机标签上的 Cross Entropy。
    loss = ce(logits, y_rand_train)
    # Backward。
    loss.backward()
    # 更新参数。
    random_optimizer.step()

    # 评估 train / val accuracy。
    random_model.eval()
    # 关闭 Gradient。
    with torch.no_grad():
        # 训练准确率。
        train_acc = float((random_model(x_rand_train).argmax(dim=1) == y_rand_train).float().mean())
        # 验证准确率。
        val_acc = float((random_model(x_rand_val).argmax(dim=1) == y_rand_val).float().mean())
    # 记录训练 Loss。
    random_loss_curve.append(float(loss.detach()))
    # 记录训练准确率。
    train_acc_curve.append(train_acc)
    # 记录验证准确率。
    val_acc_curve.append(val_acc)

# 打印最终结果。
print(f"Random labels final train loss = {random_loss_curve[-1]:.6f}")
print(f"Random labels final train acc  = {train_acc_curve[-1]:.4f}")
print(f"Random labels final val acc    = {val_acc_curve[-1]:.4f}")
print("随机标签没有可泛化规律；高训练准确率只说明模型记住了训练样本。")

# 绘制随机标签训练与验证准确率。
fig, ax = plt.subplots(figsize=(8.5, 5.2))
# 训练准确率曲线。
ax.plot(train_acc_curve, label="Train Accuracy")
# 验证准确率曲线。
ax.plot(val_acc_curve, label="Validation Accuracy")
# 二分类随机猜测参考线。
ax.axhline(0.5, linestyle="--", linewidth=1.2, label="随机猜测约 50%")
# 横轴。
ax.set_xlabel("Epoch")
# 纵轴。
ax.set_ylabel("Accuracy")
# 限定可读范围。
ax.set_ylim(0.35, 1.02)
# 标题。
ax.set_title("Random Labels：训练集可以被记住，但验证集仍接近随机猜测")
# 图例。
ax.legend()
# 网格。
ax.grid(alpha=0.2)
# 布局。
fig.tight_layout()
# 保存。
fig.savefig(FIG_DIR / "lesson41_random_labels_memorization.png", dpi=180)
# 关闭。
plt.close(fig)


# -----------------------------------------------------------------------------
# 实验5：总结表——不要只看 Training Loss
# -----------------------------------------------------------------------------
section("实验5：总结与科研解释边界")

# 打印一张简洁总结表。
print("模型      final_train   final_val    best_val    best_test   final_test")
for kind in ["linear", "small", "large"]:
    r = capacity_results[kind]
    print(
        f"{kind:6s}  {r['final_train']:11.6f}  {r['final_val']:10.6f}  "
        f"{r['best_val']:10.6f}  {r['best_test']:10.6f}  {r['final_test']:10.6f}"
    )

print("\n本讲实验结论边界：")
print("1) Train Loss 低只说明训练集拟合好，不自动等于 Generalization 好。")
print("2) Validation 用于模型选择/调参；Test 应尽量只在最后做一次独立评估。")
print("3) Generalization Gap 必须在同一指标、同一数据处理口径下解释。")
print("4) Early Stopping 是一种模型选择策略，不是证明模型已经找到真实因果规律。")
print("5) 单个合成实验只验证机制，不代表所有模型/数据集都呈现完全相同曲线。")

print("\n所有实验已完成。图像保存在:", FIG_DIR)
