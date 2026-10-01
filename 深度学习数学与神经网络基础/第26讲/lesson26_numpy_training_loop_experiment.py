# -*- coding: utf-8 -*-
"""
第26讲：NumPy 手写完整 Backpropagation + Training Loop
目标：不用 PyTorch，完成两层 MLP 的 Forward、MSE、Backward、Gradient Check、参数更新和训练循环。
"""

import numpy as np  # 导入 NumPy，用于矩阵运算
import matplotlib.pyplot as plt  # 导入 Matplotlib，用于画训练曲线
from pathlib import Path  # 导入 Path，便于管理输出路径

# -----------------------------
# 0. 基本设置
# -----------------------------
SEED = 42  # 固定随机种子，保证实验可复现
HIDDEN = 8  # 隐藏层神经元数量
MAIN_LR = 0.03  # 主实验学习率
STEPS = 2000  # 主实验训练步数
EPS = 1e-5  # 数值梯度检查使用的微小扰动

SCRIPT_DIR = Path(__file__).resolve().parent  # 获取当前脚本所在目录
FIG_DIR = SCRIPT_DIR / "figures"  # 设置实验图片输出目录
FIG_DIR.mkdir(exist_ok=True)  # 如果图片目录不存在则创建


def make_dataset():
    """创建一个最小可解释回归数据集：y = 2x + 1。"""
    x = np.linspace(-1.0, 1.0, 20, dtype=np.float64).reshape(-1, 1)  # 创建 20 个输入，Shape=(20,1)
    y = 2.0 * x + 1.0  # 按 y=2x+1 生成监督目标，Shape=(20,1)
    return x, y  # 返回输入和目标


def init_params(seed=SEED):
    """初始化 1->8->1 两层 MLP 的参数。"""
    rng = np.random.default_rng(seed)  # 创建固定种子的随机数生成器
    w1 = rng.standard_normal((1, HIDDEN)) * np.sqrt(2.0)  # 初始化第一层权重，Shape=(1,8)
    b1 = np.full((1, HIDDEN), 0.05, dtype=np.float64)  # 初始化第一层偏置，避免大量点刚好落在 ReLU=0
    w2 = rng.standard_normal((HIDDEN, 1)) * np.sqrt(2.0 / HIDDEN)  # 初始化第二层权重，Shape=(8,1)
    b2 = np.zeros((1, 1), dtype=np.float64)  # 初始化第二层偏置，Shape=(1,1)
    return w1, b1, w2, b2  # 返回四组参数


def relu(z):
    """逐元素 ReLU。"""
    return np.maximum(0.0, z)  # 负数变 0，正数保持不变


def forward(x, w1, b1, w2, b2):
    """完成两层 MLP 的 Forward，并保存 Backward 需要的中间量。"""
    z1 = x @ w1 + b1  # 第一层 Linear，Shape: (N,1)@(1,8)+(1,8) -> (N,8)
    a1 = relu(z1)  # 第一层激活，Shape 保持 (N,8)
    pred = a1 @ w2 + b2  # 第二层 Linear，Shape: (N,8)@(8,1)+(1,1) -> (N,1)
    cache = (x, z1, a1, w1, w2)  # 保存 Backward 所需中间量
    return pred, cache  # 返回预测值和缓存


def mse_loss(pred, target):
    """计算 Mean Squared Error。"""
    error = pred - target  # 计算逐样本预测误差
    loss = np.mean(error ** 2)  # 对平方误差求平均，得到标量 Loss
    return float(loss)  # 转成普通 Python 浮点数返回


def backward(pred, target, cache):
    """手写两层 MLP 的 Backward。"""
    x, z1, a1, w1, w2 = cache  # 取出 Forward 保存的中间变量
    n = x.shape[0]  # 取得 Batch 中样本数量 N

    d_pred = 2.0 * (pred - target) / n  # MSE 对预测值的梯度，Shape=(N,1)
    d_w2 = a1.T @ d_pred  # 第二层权重梯度，Shape=(8,N)@(N,1)->(8,1)
    d_b2 = np.sum(d_pred, axis=0, keepdims=True)  # 第二层偏置梯度，Shape=(1,1)
    d_a1 = d_pred @ w2.T  # 把梯度继续传回隐藏层激活，Shape=(N,1)@(1,8)->(N,8)
    d_z1 = d_a1 * (z1 > 0.0)  # 乘 ReLU 局部导数，Shape=(N,8)
    d_w1 = x.T @ d_z1  # 第一层权重梯度，Shape=(1,N)@(N,8)->(1,8)
    d_b1 = np.sum(d_z1, axis=0, keepdims=True)  # 第一层偏置梯度，Shape=(1,8)

    grads = (d_w1, d_b1, d_w2, d_b2)  # 把四组梯度打包
    return grads  # 返回全部梯度


def gradient_norm(grads):
    """把所有参数梯度合成一个整体 L2 norm，便于观察训练动态。"""
    total = sum(np.sum(g ** 2) for g in grads)  # 对所有梯度元素平方后求和
    return float(np.sqrt(total))  # 开平方得到总梯度范数


def update_params(params, grads, lr):
    """执行最基础的 Gradient Descent 参数更新。"""
    w1, b1, w2, b2 = params  # 解包参数
    d_w1, d_b1, d_w2, d_b2 = grads  # 解包梯度
    w1 = w1 - lr * d_w1  # 第一层权重沿负梯度方向更新
    b1 = b1 - lr * d_b1  # 第一层偏置沿负梯度方向更新
    w2 = w2 - lr * d_w2  # 第二层权重沿负梯度方向更新
    b2 = b2 - lr * d_b2  # 第二层偏置沿负梯度方向更新
    return w1, b1, w2, b2  # 返回更新后的参数


def loss_with_params(x, y, params):
    """给定参数，单独计算 Loss，供数值梯度检查使用。"""
    pred, _ = forward(x, *params)  # 用当前参数执行 Forward
    return mse_loss(pred, y)  # 返回 MSE


def numerical_grad_one(x, y, params, param_index, element_index, eps=EPS):
    """只对某一个参数元素做中央差分数值梯度检查。"""
    plus = [p.copy() for p in params]  # 复制一份参数，作为 +eps 版本
    minus = [p.copy() for p in params]  # 复制一份参数，作为 -eps 版本
    plus[param_index][element_index] += eps  # 对目标参数元素增加 eps
    minus[param_index][element_index] -= eps  # 对目标参数元素减小 eps
    loss_plus = loss_with_params(x, y, tuple(plus))  # 计算 +eps 后的 Loss
    loss_minus = loss_with_params(x, y, tuple(minus))  # 计算 -eps 后的 Loss
    return (loss_plus - loss_minus) / (2.0 * eps)  # 中央差分近似导数


def gradient_check(x, y, params):
    """检查 W1、b1、W2、b2 各一个元素。"""
    pred, cache = forward(x, *params)  # 执行一次 Forward
    grads = backward(pred, y, cache)  # 得到解析梯度
    checks = [  # 指定四个要检查的参数元素
        ("W1[0,0]", 0, (0, 0)),  # 检查第一层第一个 Weight
        ("b1[0,0]", 1, (0, 0)),  # 检查第一层第一个 Bias
        ("W2[0,0]", 2, (0, 0)),  # 检查第二层第一个 Weight
        ("b2[0,0]", 3, (0, 0)),  # 检查输出层 Bias
    ]
    results = []  # 用于保存检查结果
    for name, param_index, element_index in checks:  # 依次检查四个参数元素
        analytic = float(grads[param_index][element_index])  # 取手写 Backward 的解析梯度
        numeric = float(numerical_grad_one(x, y, params, param_index, element_index))  # 计算数值梯度
        error = abs(analytic - numeric)  # 计算两者绝对误差
        results.append((name, analytic, numeric, error))  # 保存结果
    return results  # 返回全部梯度检查结果


def train(x, y, lr=MAIN_LR, steps=STEPS, seed=SEED):
    """执行完整 Training Loop。"""
    params = init_params(seed)  # 使用相同随机种子初始化参数
    losses = []  # 保存每一步的 Loss
    grad_norms = []  # 保存每一步的梯度范数

    pred0, _ = forward(x, *params)  # 训练前先计算一次预测
    pred_before = pred0.copy()  # 保存训练前预测，供画图比较

    for step in range(steps):  # 重复执行固定数量的训练 step
        pred, cache = forward(x, *params)  # 1. Forward：得到预测
        loss = mse_loss(pred, y)  # 2. Loss：衡量预测误差
        grads = backward(pred, y, cache)  # 3. Backward：手工计算所有参数梯度
        losses.append(loss)  # 记录本步 Loss
        grad_norms.append(gradient_norm(grads))  # 记录本步总梯度范数
        params = update_params(params, grads, lr)  # 4. Update：沿负梯度更新参数

    pred_after, _ = forward(x, *params)  # 训练结束后重新计算预测
    return params, np.array(losses), np.array(grad_norms), pred_before, pred_after  # 返回训练结果


def run_lr_comparison(x, y):
    """固定相同初始化，只改变 Learning Rate。"""
    configs = [(0.003, 500), (0.03, 500), (1.0, 80)]  # 设置慢、合适、过大三组学习率
    histories = {}  # 保存每组 Loss 曲线
    for lr, steps in configs:  # 逐组运行
        _, losses, _, _, _ = train(x, y, lr=lr, steps=steps, seed=SEED)  # 保持相同 seed 训练
        histories[lr] = losses  # 保存对应曲线
    return histories  # 返回三组结果


def save_figures(x, y, losses, grad_norms, pred_before, pred_after, lr_histories):
    """保存课程需要的四张实验图。"""
    plt.figure(figsize=(7, 4.5))  # 创建训练 Loss 曲线画布
    plt.plot(np.arange(len(losses)), losses)  # 绘制每一步 Loss
    plt.xlabel("Training step")  # 设置横轴名称
    plt.ylabel("MSE loss")  # 设置纵轴名称
    plt.title("Manual NumPy MLP: training loss")  # 设置图标题
    plt.yscale("log")  # 使用对数纵轴，便于观察多数量级下降
    plt.grid(True, alpha=0.3)  # 添加浅色网格
    plt.tight_layout()  # 自动调整布局
    plt.savefig(FIG_DIR / "lesson26_training_loss.png", dpi=180)  # 保存图片
    plt.close()  # 关闭当前画布

    plt.figure(figsize=(7, 4.5))  # 创建预测前后对比画布
    plt.plot(x[:, 0], y[:, 0], label="Target y=2x+1")  # 绘制真实目标
    plt.scatter(x[:, 0], pred_before[:, 0], label="Before training")  # 绘制训练前预测
    plt.scatter(x[:, 0], pred_after[:, 0], label="After training")  # 绘制训练后预测
    plt.xlabel("x")  # 设置横轴名称
    plt.ylabel("y")  # 设置纵轴名称
    plt.title("Predictions before and after training")  # 设置标题
    plt.legend()  # 显示图例
    plt.grid(True, alpha=0.3)  # 添加浅色网格
    plt.tight_layout()  # 自动调整布局
    plt.savefig(FIG_DIR / "lesson26_before_after_fit.png", dpi=180)  # 保存图片
    plt.close()  # 关闭当前画布

    plt.figure(figsize=(7, 4.5))  # 创建梯度范数画布
    plt.plot(np.arange(len(grad_norms)), grad_norms)  # 绘制梯度范数随 step 的变化
    plt.xlabel("Training step")  # 设置横轴名称
    plt.ylabel("Global gradient norm")  # 设置纵轴名称
    plt.title("Gradient norm during training")  # 设置标题
    plt.yscale("log")  # 使用对数纵轴
    plt.grid(True, alpha=0.3)  # 添加浅色网格
    plt.tight_layout()  # 自动调整布局
    plt.savefig(FIG_DIR / "lesson26_gradient_norm.png", dpi=180)  # 保存图片
    plt.close()  # 关闭当前画布

    plt.figure(figsize=(7, 4.5))  # 创建 Learning Rate 对比画布
    for lr, history in lr_histories.items():  # 逐个学习率绘图
        plt.plot(np.arange(len(history)), history, label=f"lr={lr}")  # 绘制对应 Loss 曲线
    plt.xlabel("Training step")  # 设置横轴名称
    plt.ylabel("MSE loss")  # 设置纵轴名称
    plt.title("Same initialization, different learning rates")  # 设置标题
    plt.yscale("log")  # 使用对数纵轴
    plt.legend()  # 显示图例
    plt.grid(True, alpha=0.3)  # 添加浅色网格
    plt.tight_layout()  # 自动调整布局
    plt.savefig(FIG_DIR / "lesson26_learning_rate_comparison.png", dpi=180)  # 保存图片
    plt.close()  # 关闭当前画布


def main():
    """主程序：按实验设计依次执行所有验证。"""
    np.set_printoptions(precision=6, suppress=True)  # 设置 NumPy 输出格式，便于阅读
    x, y = make_dataset()  # 创建训练数据
    initial_params = init_params(SEED)  # 初始化一份参数，供梯度检查

    print("=" * 72)  # 打印分隔线
    print("第26讲：NumPy 手写完整 Backpropagation + Training Loop")  # 打印实验标题
    print("=" * 72)  # 打印分隔线
    print(f"X.shape={x.shape}, Y.shape={y.shape}")  # 输出数据 Shape
    print(f"网络结构: 1 -> {HIDDEN} -> 1")  # 输出网络结构
    print(f"主实验: learning_rate={MAIN_LR}, steps={STEPS}, seed={SEED}")  # 输出关键实验设置

    print("\n[实验1] 训练前的 Gradient Check")  # 打印实验1标题
    checks = gradient_check(x, y, initial_params)  # 执行数值梯度检查
    for name, analytic, numeric, error in checks:  # 输出每个被检查参数的结果
        print(f"{name:8s} analytic={analytic:+.9f} numeric={numeric:+.9f} abs_error={error:.3e}")  # 打印解析/数值梯度

    print("\n[实验2] 完整 Training Loop")  # 打印实验2标题
    params, losses, grad_norms, pred_before, pred_after = train(x, y)  # 执行主训练实验
    checkpoints = [0, 1, 10, 50, 100, 500, 1000, STEPS - 1]  # 选择代表性的训练 step
    for step in checkpoints:  # 输出这些 step 的训练状态
        print(f"step={step:4d} loss={losses[step]:.9f} grad_norm={grad_norms[step]:.9f}")  # 打印 Loss 和梯度范数

    final_mae = float(np.mean(np.abs(pred_after - y)))  # 计算最终平均绝对误差
    final_max_error = float(np.max(np.abs(pred_after - y)))  # 计算最终最大绝对误差
    print(f"final_mae={final_mae:.9f}")  # 输出最终 MAE
    print(f"final_max_abs_error={final_max_error:.9f}")  # 输出最终最大绝对误差
    print("前5个目标值:", y[:5, 0])  # 输出前5个目标
    print("前5个训练前预测:", pred_before[:5, 0])  # 输出前5个训练前预测
    print("前5个训练后预测:", pred_after[:5, 0])  # 输出前5个训练后预测

    print("\n[实验3] 固定相同初始化，只比较 Learning Rate")  # 打印实验3标题
    lr_histories = run_lr_comparison(x, y)  # 运行学习率对照实验
    for lr, history in lr_histories.items():  # 输出每个学习率的结果
        print(f"lr={lr:<5} steps={len(history):4d} initial_loss={history[0]:.9f} final_loss={history[-1]:.9f} min_loss={history.min():.9f}")  # 打印对照结果

    print("\n[实验4] 保存训练曲线与结果图")  # 打印实验4标题
    save_figures(x, y, losses, grad_norms, pred_before, pred_after, lr_histories)  # 保存四张实验图
    print("已生成:")  # 输出生成图片清单
    for path in sorted(FIG_DIR.glob("lesson26_*.png")):  # 遍历课程图片
        print(" -", path.name)  # 打印每张图片名称

    print("\n[科研边界] 本实验只说明训练集上的优化闭环能够工作。")  # 强调实验结论边界
    print("它没有证明模型对新数据的泛化能力；Train/Validation/Test 会在后续课程系统学习。")  # 强调不能过度解释


if __name__ == "__main__":  # 只有直接运行脚本时才执行 main
    main()  # 启动完整实验
