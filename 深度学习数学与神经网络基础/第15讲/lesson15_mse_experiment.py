# 第15讲实验：MSE - 为什么要平方？为什么要平均？
# 运行方式：python lesson15_mse_experiment.py

# 导入 NumPy，用于数组和数值计算。
import numpy as np
# 导入 Matplotlib，用于画实验结果图。
import matplotlib.pyplot as plt
# 导入 Path，用于可靠地创建和定位输出文件夹。
from pathlib import Path

# 取得当前脚本所在目录。
BASE_DIR = Path(__file__).resolve().parent
# 定义实验图保存目录。
FIG_DIR = BASE_DIR / "figures"
# 如果 figures 目录不存在，就自动创建。
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 设置 NumPy 打印格式，便于阅读小数。
np.set_printoptions(precision=4, suppress=True)

# 定义一个辅助函数，用来打印明显的实验标题。
def section(title):
    # 打印空行和分隔线。
    print("\n" + "=" * 72)
    # 打印当前实验标题。
    print(title)
    # 再打印一条分隔线。
    print("=" * 72)

# 定义逐元素平方误差函数。
def squared_error(prediction, target):
    # prediction - target 得到带正负号的误差。
    error = prediction - target
    # 对误差逐元素平方，消除正负抵消并放大大误差。
    return error ** 2

# 定义 MSE 函数。
def mse(prediction, target):
    # 先计算所有元素的平方误差。
    se = squared_error(prediction, target)
    # 再对所有平方误差取平均，得到一个标量 MSE。
    return np.mean(se)

# 定义 MAE，仅用于和 MSE 做对照，不作为本讲主角。
def mae(prediction, target):
    # 取绝对误差，再求平均。
    return np.mean(np.abs(prediction - target))

# -----------------------------------------------------------------------------
# 实验1：为什么不能直接平均带符号 Error？为什么平方以后不会抵消？
# -----------------------------------------------------------------------------
section("实验1：正负 Error 会抵消，但 Squared Error 不会")
# 构造三个预测值。
pred_1 = np.array([2.0, 4.0, 6.0])
# 三个样本的真实值都设为 4。
target_1 = np.array([4.0, 4.0, 4.0])
# 计算带符号误差。
error_1 = pred_1 - target_1
# 计算逐元素平方误差。
se_1 = squared_error(pred_1, target_1)
# 计算普通带符号误差的平均值。
mean_error_1 = np.mean(error_1)
# 计算 MSE。
mse_1 = np.mean(se_1)
# 打印预测值。
print("Prediction:", pred_1)
# 打印真实值。
print("Target:    ", target_1)
# 打印带符号误差。
print("Error:     ", error_1)
# 打印平方误差。
print("SquaredErr:", se_1)
# 打印平均带符号误差。
print(f"Mean(Error) = {mean_error_1:.4f}")
# 打印 MSE。
print(f"MSE         = {mse_1:.4f}")

# -----------------------------------------------------------------------------
# 实验2：完整手算一个 Batch 的 MSE。
# -----------------------------------------------------------------------------
section("实验2：手算一个 Batch 的 MSE")
# 构造 4 个样本的预测值。
pred_2 = np.array([3.0, 5.0, 2.0, 8.0])
# 构造 4 个样本的真实值。
target_2 = np.array([4.0, 4.0, 2.0, 6.0])
# 计算误差。
error_2 = pred_2 - target_2
# 计算平方误差。
se_2 = error_2 ** 2
# 计算平方误差总和。
sse_2 = np.sum(se_2)
# 计算平均平方误差。
mse_2 = np.mean(se_2)
# 打印预测值。
print("Prediction:", pred_2)
# 打印真实值。
print("Target:    ", target_2)
# 打印误差。
print("Error:     ", error_2)
# 打印平方误差。
print("SquaredErr:", se_2)
# 打印平方误差之和。
print(f"Sum of Squared Error = {sse_2:.4f}")
# 打印 MSE。
print(f"MSE                  = {mse_2:.4f}")

# -----------------------------------------------------------------------------
# 实验3：为什么常见训练 Loss 使用 mean reduction，而不是直接 sum？
# -----------------------------------------------------------------------------
section("实验3：Mean 与 Sum - 复制 Batch 后会发生什么")
# 计算原 Batch 的平方误差总和。
sum_original = np.sum(se_2)
# 计算原 Batch 的平方误差平均值。
mean_original = np.mean(se_2)
# 把完全相同的预测 Batch 复制一遍，Batch Size 翻倍。
pred_duplicated = np.concatenate([pred_2, pred_2])
# 把真实值也复制一遍。
target_duplicated = np.concatenate([target_2, target_2])
# 计算复制后 Batch 的平方误差。
se_duplicated = (pred_duplicated - target_duplicated) ** 2
# 计算复制后的 sum。
sum_duplicated = np.sum(se_duplicated)
# 计算复制后的 mean。
mean_duplicated = np.mean(se_duplicated)
# 打印原 Batch Size。
print("原 Batch Size:", pred_2.size)
# 打印复制后的 Batch Size。
print("复制后 Batch Size:", pred_duplicated.size)
# 打印原 sum。
print(f"原 Sum  = {sum_original:.4f}")
# 打印复制后 sum。
print(f"新 Sum  = {sum_duplicated:.4f}")
# 打印原 mean。
print(f"原 Mean = {mean_original:.4f}")
# 打印复制后 mean。
print(f"新 Mean = {mean_duplicated:.4f}")

# -----------------------------------------------------------------------------
# 实验4：二维输出时，MSE 到底在对哪些元素求平均？
# -----------------------------------------------------------------------------
section("实验4：多输出 Shape 下的逐元素 MSE 与 reduction")
# 构造 Shape=(2,3) 的预测矩阵：2 个样本，每个样本 3 个输出。
pred_4 = np.array([[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0]])
# 构造同 Shape 的真实矩阵。
target_4 = np.array([[1.0, 1.0, 5.0],
                     [5.0, 5.0, 4.0]])
# 逐元素计算 Error。
error_4 = pred_4 - target_4
# 逐元素平方。
se_4 = error_4 ** 2
# 沿最后一维求平均，得到每个样本自己的 MSE，Shape=(2,)。
per_sample_mse_4 = np.mean(se_4, axis=1)
# 对全部 6 个元素统一求平均，得到全局标量 MSE。
global_mse_4 = np.mean(se_4)
# 打印预测矩阵及 Shape。
print("Prediction shape:", pred_4.shape)
print(pred_4)
# 打印 Target。
print("Target shape:", target_4.shape)
print(target_4)
# 打印逐元素平方误差。
print("Squared Error:")
print(se_4)
# 打印每样本 MSE。
print("Per-sample MSE:", per_sample_mse_4, "shape =", per_sample_mse_4.shape)
# 打印全局 MSE。
print(f"Global MSE = {global_mse_4:.4f}")

# -----------------------------------------------------------------------------
# 实验5：MSE 为什么对大误差特别敏感？用 MAE 做对照。
# -----------------------------------------------------------------------------
section("实验5：一个大误差怎样主导 MSE")
# 构造没有异常大误差的一组 Error。
error_normal = np.array([1.0, 1.0, 1.0, 1.0])
# 构造只有最后一个样本出现大误差的一组 Error。
error_outlier = np.array([1.0, 1.0, 1.0, 10.0])
# 用全 0 Target，把 Error 本身当作 Prediction，方便直接比较。
target_zero_normal = np.zeros_like(error_normal)
# 为第二组构造全 0 Target。
target_zero_outlier = np.zeros_like(error_outlier)
# 计算正常组 MSE。
mse_normal = mse(error_normal, target_zero_normal)
# 计算异常组 MSE。
mse_outlier = mse(error_outlier, target_zero_outlier)
# 计算正常组 MAE。
mae_normal = mae(error_normal, target_zero_normal)
# 计算异常组 MAE。
mae_outlier = mae(error_outlier, target_zero_outlier)
# 打印两组 Error。
print("普通误差:", error_normal)
print("含大误差:", error_outlier)
# 打印 MSE 对比。
print(f"MSE: 普通={mse_normal:.4f}, 含大误差={mse_outlier:.4f}")
# 打印 MAE 对比。
print(f"MAE: 普通={mae_normal:.4f}, 含大误差={mae_outlier:.4f}")

# -----------------------------------------------------------------------------
# 实验6：把 MSE 看成参数 w 的函数，观察 Loss Landscape。
# -----------------------------------------------------------------------------
section("实验6：MSE 作为参数 w 的函数")
# 固定一个输入 x=2。
x_value = 2.0
# 固定真实答案 y=6。
y_target = 6.0
# 生成一系列候选参数 w。
w_values = np.linspace(0.0, 6.0, 121)
# 对每个 w 计算 prediction = w*x。
pred_values = w_values * x_value
# 因为这里只有一个样本，所以 MSE 就是单个平方误差。
loss_values = (pred_values - y_target) ** 2
# 找到最小 Loss 所在的位置。
best_index = np.argmin(loss_values)
# 取出最优 w。
best_w = w_values[best_index]
# 取出最小 MSE。
best_loss = loss_values[best_index]
# 打印结果。
print(f"最优 w ≈ {best_w:.4f}")
print(f"最小 MSE ≈ {best_loss:.4f}")

# -----------------------------------------------------------------------------
# 实验7：用数值导数预告下一阶段 - Loss 对 w 的敏感度。
# -----------------------------------------------------------------------------
section("实验7：数值导数预告 - 当前 w 改一点，MSE 会怎样变化")
# 定义一个只依赖 w 的 Loss 函数。
def loss_of_w(w):
    # 先根据 w 计算预测。
    prediction = w * x_value
    # 返回单样本 MSE。
    return (prediction - y_target) ** 2

# 选择当前参数 w=2。
w0 = 2.0
# 选择一个很小的扰动 h。
h = 1e-5
# 用中心差分估计 dL/dw。
numeric_grad = (loss_of_w(w0 + h) - loss_of_w(w0 - h)) / (2.0 * h)
# 这个例子的解析导数为 2*(wx-y)*x。
analytic_grad = 2.0 * (w0 * x_value - y_target) * x_value
# 打印当前 Loss。
print(f"w = {w0:.4f}, MSE = {loss_of_w(w0):.4f}")
# 打印数值导数。
print(f"数值估计 dL/dw = {numeric_grad:.6f}")
# 打印解析导数。
print(f"解析结果 dL/dw = {analytic_grad:.6f}")
# 根据负梯度方向给出方向判断。
print("因为 dL/dw < 0，所以要让 Loss 下降，w 应该向更大的方向移动。")

# -----------------------------------------------------------------------------
# 实验8：如果安装了 PyTorch，就验证 nn.MSELoss 与 NumPy 一致。
# -----------------------------------------------------------------------------
section("实验8：NumPy 与 PyTorch MSELoss 对照（如果已安装 PyTorch）")
# 用 try 尝试导入 PyTorch，这样没安装时脚本也能继续完成前面实验。
try:
    # 导入 PyTorch。
    import torch
    # 导入 torch.nn。
    import torch.nn as nn
    # 把 NumPy 预测值转换成 PyTorch Tensor。
    pred_t = torch.tensor(pred_2, dtype=torch.float32)
    # 把 NumPy Target 转换成 PyTorch Tensor。
    target_t = torch.tensor(target_2, dtype=torch.float32)
    # 创建默认 reduction='mean' 的 MSELoss。
    criterion_mean = nn.MSELoss(reduction="mean")
    # 创建 reduction='sum' 的 MSELoss。
    criterion_sum = nn.MSELoss(reduction="sum")
    # 计算 PyTorch mean MSE。
    torch_mean = criterion_mean(pred_t, target_t).item()
    # 计算 PyTorch sum squared error。
    torch_sum = criterion_sum(pred_t, target_t).item()
    # 打印 NumPy 与 PyTorch 对照。
    print(f"NumPy mean MSE = {mse_2:.4f}")
    print(f"PyTorch mean   = {torch_mean:.4f}")
    print(f"NumPy sum SE   = {sse_2:.4f}")
    print(f"PyTorch sum    = {torch_sum:.4f}")
# 如果 PyTorch 不可用，则给出说明但不让程序报错退出。
except Exception as exc:
    # 打印跳过信息。
    print("未完成 PyTorch 对照，原因：", repr(exc))

# -----------------------------------------------------------------------------
# 图1：单个 Error 的绝对值与平方惩罚。
# -----------------------------------------------------------------------------
# 生成从 -4 到 4 的连续误差。
error_axis = np.linspace(-4.0, 4.0, 401)
# 计算 MAE 单点惩罚 |e|，仅作对照。
mae_curve = np.abs(error_axis)
# 计算 MSE 单点惩罚 e^2。
mse_curve = error_axis ** 2
# 创建一张新图。
plt.figure(figsize=(8, 5))
# 绘制绝对误差曲线。
plt.plot(error_axis, mae_curve, label="Absolute error |e|")
# 绘制平方误差曲线。
plt.plot(error_axis, mse_curve, label="Squared error e^2")
# 标记横轴名称。
plt.xlabel("Error e = prediction - target")
# 标记纵轴名称。
plt.ylabel("Penalty")
# 设置图标题。
plt.title("Squared error grows much faster for large errors")
# 显示图例。
plt.legend()
# 添加网格便于观察。
plt.grid(alpha=0.3)
# 自动整理布局。
plt.tight_layout()
# 保存图像。
plt.savefig(FIG_DIR / "lesson15_squared_vs_absolute_error.png", dpi=180)
# 关闭当前图，避免影响下一张图。
plt.close()

# -----------------------------------------------------------------------------
# 图2：比较正常误差和一个异常大误差对 MSE / MAE 的影响。
# -----------------------------------------------------------------------------
# 组织柱状图需要的数据。
metrics_normal = [mse_normal, mae_normal]
# 组织含大误差时的数据。
metrics_outlier = [mse_outlier, mae_outlier]
# 创建 x 轴位置。
x_pos = np.arange(2)
# 设置柱宽。
width = 0.34
# 创建新图。
plt.figure(figsize=(8, 5))
# 画普通误差组。
plt.bar(x_pos - width / 2, metrics_normal, width, label="Errors [1,1,1,1]")
# 画含大误差组。
plt.bar(x_pos + width / 2, metrics_outlier, width, label="Errors [1,1,1,10]")
# 设置 x 轴标签。
plt.xticks(x_pos, ["MSE", "MAE"])
# 设置 y 轴标签。
plt.ylabel("Loss value")
# 设置标题。
plt.title("One large error affects MSE much more strongly")
# 显示图例。
plt.legend()
# 添加横向网格。
plt.grid(axis="y", alpha=0.3)
# 自动整理布局。
plt.tight_layout()
# 保存图像。
plt.savefig(FIG_DIR / "lesson15_outlier_sensitivity.png", dpi=180)
# 关闭图像。
plt.close()

# -----------------------------------------------------------------------------
# 图3：MSE 随参数 w 的变化。
# -----------------------------------------------------------------------------
# 创建新图。
plt.figure(figsize=(8, 5))
# 绘制 w 与 MSE 的关系。
plt.plot(w_values, loss_values)
# 标记最优点。
plt.scatter([best_w], [best_loss], s=60, label=f"minimum at w={best_w:.1f}")
# 设置 x 轴。
plt.xlabel("Parameter w")
# 设置 y 轴。
plt.ylabel("MSE")
# 设置标题。
plt.title("Loss landscape for prediction = w * 2, target = 6")
# 显示图例。
plt.legend()
# 添加网格。
plt.grid(alpha=0.3)
# 自动整理布局。
plt.tight_layout()
# 保存图像。
plt.savefig(FIG_DIR / "lesson15_weight_mse_landscape.png", dpi=180)
# 关闭图像。
plt.close()

# 打印所有图像保存位置。
section("图像输出")
# 打印第一张图路径。
print(FIG_DIR / "lesson15_squared_vs_absolute_error.png")
# 打印第二张图路径。
print(FIG_DIR / "lesson15_outlier_sensitivity.png")
# 打印第三张图路径。
print(FIG_DIR / "lesson15_weight_mse_landscape.png")
