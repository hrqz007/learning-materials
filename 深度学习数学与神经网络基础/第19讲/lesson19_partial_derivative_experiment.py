# 第19讲实验：Partial Derivative（偏导数）
# 目标：用数值差分验证“只改变一个变量、其他变量保持不变”的偏导数含义。

import numpy as np  # 导入 NumPy，用于数值计算
import matplotlib.pyplot as plt  # 导入 Matplotlib，用于绘图
from matplotlib import font_manager  # 导入字体管理器，保证中文标签可显示

# 注册中文字体，避免绘图时中文变成方块。
font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"  # 指定系统中文字体路径
font_manager.fontManager.addfont(font_path)  # 把字体加入 Matplotlib 字体管理器
plt.rcParams["font.family"] = "Noto Sans CJK JP"  # 设置默认中文字体
plt.rcParams["axes.unicode_minus"] = False  # 允许坐标轴正确显示负号


def central_partial_x(func, x, y, eps=1e-5):
    """数值估计 ∂f/∂x：只改变 x，保持 y 不变。"""
    return (func(x + eps, y) - func(x - eps, y)) / (2.0 * eps)  # 使用中心差分提高精度


def central_partial_y(func, x, y, eps=1e-5):
    """数值估计 ∂f/∂y：只改变 y，保持 x 不变。"""
    return (func(x, y + eps) - func(x, y - eps)) / (2.0 * eps)  # 使用中心差分提高精度


def toy_function(x, y):
    """二维函数 f(x,y)=x^2+3xy+y^2。"""
    return x**2 + 3.0 * x * y + y**2  # 计算函数值


def linear_prediction(w1, w2, b, x1, x2):
    """两特征线性模型：ŷ=w1*x1+w2*x2+b。"""
    return w1 * x1 + w2 * x2 + b  # 返回模型预测


def squared_loss(w1, w2, b, x1, x2, target):
    """单样本平方损失 L=(ŷ-y)^2。"""
    y_hat = linear_prediction(w1, w2, b, x1, x2)  # 先得到预测值
    return (y_hat - target) ** 2  # 再计算平方误差


def numeric_partial_parameter(name, w1, w2, b, x1, x2, target, eps=1e-5):
    """数值估计损失对 w1、w2 或 b 的偏导数。"""
    if name == "w1":  # 如果要估计对 w1 的偏导
        plus = squared_loss(w1 + eps, w2, b, x1, x2, target)  # 只把 w1 增加 eps
        minus = squared_loss(w1 - eps, w2, b, x1, x2, target)  # 只把 w1 减少 eps
    elif name == "w2":  # 如果要估计对 w2 的偏导
        plus = squared_loss(w1, w2 + eps, b, x1, x2, target)  # 只改变 w2
        minus = squared_loss(w1, w2 - eps, b, x1, x2, target)  # 其他量保持不变
    elif name == "b":  # 如果要估计对偏置 b 的偏导
        plus = squared_loss(w1, w2, b + eps, x1, x2, target)  # 只改变 b
        minus = squared_loss(w1, w2, b - eps, x1, x2, target)  # 其他量保持不变
    else:  # 如果名字不是三种允许的参数
        raise ValueError("name 必须是 w1、w2 或 b")  # 主动报错避免静默错误
    return (plus - minus) / (2.0 * eps)  # 用中心差分得到偏导数估计


print("=" * 72)  # 打印分隔线
print("实验1：二维函数的两个偏导数")  # 输出实验标题
print("=" * 72)  # 打印分隔线
x = 2.0  # 设定当前 x
y = 1.0  # 设定当前 y
f_value = toy_function(x, y)  # 计算当前位置的函数值
analytic_dx = 2.0 * x + 3.0 * y  # 根据公式求 ∂f/∂x=2x+3y
analytic_dy = 3.0 * x + 2.0 * y  # 根据公式求 ∂f/∂y=3x+2y
numeric_dx = central_partial_x(toy_function, x, y)  # 数值估计 ∂f/∂x
numeric_dy = central_partial_y(toy_function, x, y)  # 数值估计 ∂f/∂y
print(f"点 (x,y)=({x:.1f},{y:.1f})")  # 打印当前点
print(f"f(x,y)={f_value:.6f}")  # 打印函数值
print(f"解析 ∂f/∂x={analytic_dx:.6f}，数值估计={numeric_dx:.6f}")  # 对照 x 偏导
print(f"解析 ∂f/∂y={analytic_dy:.6f}，数值估计={numeric_dy:.6f}")  # 对照 y 偏导

print("\n" + "=" * 72)  # 换行并打印分隔线
print("实验2：一个 Loss 同时受 w1、w2、b 影响")  # 输出实验标题
print("=" * 72)  # 打印分隔线
x1 = 2.0  # 第一个输入特征
x2 = -1.0  # 第二个输入特征
w1 = 1.5  # 第一个权重
w2 = -0.5  # 第二个权重
b = 0.2  # 偏置
target = 3.0  # 真实目标值
y_hat = linear_prediction(w1, w2, b, x1, x2)  # 计算当前预测
error = y_hat - target  # 计算预测误差
loss = error**2  # 计算平方损失
analytic_dw1 = 2.0 * error * x1  # 根据链式关系求 ∂L/∂w1
analytic_dw2 = 2.0 * error * x2  # 根据链式关系求 ∂L/∂w2
analytic_db = 2.0 * error  # 根据链式关系求 ∂L/∂b
numeric_dw1 = numeric_partial_parameter("w1", w1, w2, b, x1, x2, target)  # 数值估计 ∂L/∂w1
numeric_dw2 = numeric_partial_parameter("w2", w1, w2, b, x1, x2, target)  # 数值估计 ∂L/∂w2
numeric_db = numeric_partial_parameter("b", w1, w2, b, x1, x2, target)  # 数值估计 ∂L/∂b
print(f"输入 x1={x1:.1f}, x2={x2:.1f}, target={target:.1f}")  # 打印输入与标签
print(f"参数 w1={w1:.1f}, w2={w2:.1f}, b={b:.1f}")  # 打印当前参数
print(f"预测 y_hat={y_hat:.6f}，error={error:.6f}，loss={loss:.6f}")  # 打印预测和损失
print(f"∂L/∂w1：解析={analytic_dw1:.6f}，数值={numeric_dw1:.6f}")  # 打印 w1 偏导对照
print(f"∂L/∂w2：解析={analytic_dw2:.6f}，数值={numeric_dw2:.6f}")  # 打印 w2 偏导对照
print(f"∂L/∂b ：解析={analytic_db:.6f}，数值={numeric_db:.6f}")  # 打印 b 偏导对照

print("\n" + "=" * 72)  # 换行并打印分隔线
print("实验3：偏导数的正负号能否预测小扰动后的 Loss 变化？")  # 输出实验标题
print("=" * 72)  # 打印分隔线
delta = 0.01  # 设定一个很小的参数改变量
loss_w1_plus = squared_loss(w1 + delta, w2, b, x1, x2, target)  # 只增加 w1 后重新计算 loss
actual_change_w1 = loss_w1_plus - loss  # 计算真实 loss 变化
predicted_change_w1 = analytic_dw1 * delta  # 用偏导数做一阶近似预测
loss_w2_plus = squared_loss(w1, w2 + delta, b, x1, x2, target)  # 只增加 w2 后重新计算 loss
actual_change_w2 = loss_w2_plus - loss  # 计算真实 loss 变化
predicted_change_w2 = analytic_dw2 * delta  # 用偏导数预测 w2 改变后的 loss 变化
print(f"w1 增加 {delta:.2f}：预测 ΔL≈{predicted_change_w1:.6f}，实际 ΔL={actual_change_w1:.6f}")  # 输出 w1 对照
print(f"w2 增加 {delta:.2f}：预测 ΔL≈{predicted_change_w2:.6f}，实际 ΔL={actual_change_w2:.6f}")  # 输出 w2 对照
print("解释：∂L/∂w1>0，所以增加 w1 会让 Loss 上升；∂L/∂w2<0，所以增加 w2 会让 Loss 下降。")  # 输出符号直觉

print("\n" + "=" * 72)  # 换行并打印分隔线
print("实验4：二维 MSE Loss Surface 的水平/垂直切片")  # 输出实验标题
print("=" * 72)  # 打印分隔线
X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])  # 构造三个二维样本
Y = np.array([2.0, -1.0, 1.0])  # 构造目标，最优参数恰好接近 (2,-1)


def dataset_mse(a, c):
    """给定两个权重 a、c，计算三个样本上的 MSE。"""
    pred = X @ np.array([a, c])  # 使用矩阵乘法得到三个预测
    return np.mean((pred - Y) ** 2)  # 返回三个样本的平均平方误差


current_w1 = 1.0  # 选取当前 w1
current_w2 = 0.0  # 选取当前 w2
current_loss = dataset_mse(current_w1, current_w2)  # 计算当前 MSE
partial_w1 = central_partial_x(lambda a, c: dataset_mse(a, c), current_w1, current_w2)  # 数值估计对 w1 偏导
partial_w2 = central_partial_y(lambda a, c: dataset_mse(a, c), current_w1, current_w2)  # 数值估计对 w2 偏导
print(f"当前位置 (w1,w2)=({current_w1:.1f},{current_w2:.1f})，MSE={current_loss:.6f}")  # 打印当前损失
print(f"固定 w2，只看 w1 方向：∂L/∂w1≈{partial_w1:.6f}")  # 打印水平方向偏导
print(f"固定 w1，只看 w2 方向：∂L/∂w2≈{partial_w2:.6f}")  # 打印垂直方向偏导

# 绘制函数 f(x,y) 的二维等高线，展示“只沿一个轴移动”的偏导直觉。
xs = np.linspace(-3.0, 3.0, 220)  # 生成 x 网格
ys = np.linspace(-3.0, 3.0, 220)  # 生成 y 网格
XX, YY = np.meshgrid(xs, ys)  # 构造二维网格
ZZ = toy_function(XX, YY)  # 计算网格上的函数值
plt.figure(figsize=(8, 6))  # 创建图像
contour = plt.contour(XX, YY, ZZ, levels=18)  # 绘制等高线
plt.clabel(contour, inline=True, fontsize=7)  # 给等高线添加数值标签
plt.scatter([x], [y], s=50)  # 标出当前点
plt.plot([x - 1.2, x + 1.2], [y, y], linewidth=2, label="固定 y，只改变 x")  # 画 x 方向切片
plt.plot([x, x], [y - 1.2, y + 1.2], linewidth=2, label="固定 x，只改变 y")  # 画 y 方向切片
plt.xlabel("x")  # 设置横轴标签
plt.ylabel("y")  # 设置纵轴标签
plt.title("偏导数：一次只沿一个坐标轴观察局部变化")  # 设置标题
plt.legend()  # 显示图例
plt.tight_layout()  # 自动调整边距
plt.savefig("/mnt/data/deep_learning_lesson19/final/figures/lesson19_partial_axes_contour.png", dpi=180)  # 保存图片
plt.close()  # 关闭图像释放内存

# 绘制二维 MSE Loss Surface 的等高线，并标出当前点与理论最优点。
w1_values = np.linspace(-1.0, 4.0, 240)  # 生成 w1 网格
w2_values = np.linspace(-3.0, 2.0, 240)  # 生成 w2 网格
W1, W2 = np.meshgrid(w1_values, w2_values)  # 构造二维参数网格
LOSS = np.zeros_like(W1)  # 创建与网格同 Shape 的损失矩阵
for i in range(W1.shape[0]):  # 遍历每一行网格
    for j in range(W1.shape[1]):  # 遍历每一列网格
        LOSS[i, j] = dataset_mse(W1[i, j], W2[i, j])  # 计算对应参数组合的 MSE
plt.figure(figsize=(8, 6))  # 创建 Loss 等高线图
levels = np.geomspace(0.01, max(0.02, float(LOSS.max())), 20)  # 生成对数间隔等高线层级
contour2 = plt.contour(W1, W2, LOSS, levels=levels)  # 绘制 MSE 等高线
plt.scatter([current_w1], [current_w2], s=60, label="当前参数 (1,0)")  # 标出当前参数
plt.scatter([2.0], [-1.0], s=60, marker="x", label="低 Loss 区域中心 (2,-1)")  # 标出低损失位置
plt.axhline(current_w2, linewidth=1.2, linestyle="--")  # 画固定 w2 的水平切片
plt.axvline(current_w1, linewidth=1.2, linestyle="--")  # 画固定 w1 的垂直切片
plt.xlabel("w1")  # 设置横轴标签
plt.ylabel("w2")  # 设置纵轴标签
plt.title("二维 MSE Loss：偏导数分别观察 w1 与 w2 方向")  # 设置标题
plt.legend()  # 显示图例
plt.tight_layout()  # 自动调整边距
plt.savefig("/mnt/data/deep_learning_lesson19/final/figures/lesson19_mse_loss_contour.png", dpi=180)  # 保存图片
plt.close()  # 关闭图像释放内存

# 绘制一维切片，直观看“固定另一个参数”是什么意思。
w1_slice_loss = np.array([dataset_mse(v, current_w2) for v in w1_values])  # 固定 w2=0，只改变 w1
w2_slice_loss = np.array([dataset_mse(current_w1, v) for v in w2_values])  # 固定 w1=1，只改变 w2
plt.figure(figsize=(8, 5))  # 创建切片图
plt.plot(w1_values, w1_slice_loss, label="固定 w2=0，改变 w1")  # 绘制 w1 切片
plt.plot(w2_values, w2_slice_loss, label="固定 w1=1，改变 w2")  # 绘制 w2 切片
plt.xlabel("被改变的参数值")  # 设置横轴标签
plt.ylabel("MSE Loss")  # 设置纵轴标签
plt.title("Partial Derivative 的切片直觉：其他参数保持不变")  # 设置标题
plt.legend()  # 显示图例
plt.tight_layout()  # 自动调整边距
plt.savefig("/mnt/data/deep_learning_lesson19/final/figures/lesson19_loss_slices.png", dpi=180)  # 保存图片
plt.close()  # 关闭图像释放内存

print("\n已生成 3 张实验图：")  # 打印图片生成提示
print("1. lesson19_partial_axes_contour.png")  # 打印第一张图文件名
print("2. lesson19_mse_loss_contour.png")  # 打印第二张图文件名
print("3. lesson19_loss_slices.png")  # 打印第三张图文件名
