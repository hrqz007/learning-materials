# 第20讲实验：Gradient（梯度）
# 目标：把多个偏导数组成梯度向量，并验证梯度/负梯度的方向含义。

import numpy as np  # 导入 NumPy，用于向量、矩阵和数值计算
import matplotlib.pyplot as plt  # 导入 Matplotlib，用于绘制 Loss 等高线和梯度方向
from matplotlib import font_manager  # 导入字体管理器，保证中文标签正常显示

# 注册中文字体，避免图片中的中文变成方块。
font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"  # 系统中的中文字体路径
font_manager.fontManager.addfont(font_path)  # 把字体加入 Matplotlib 字体管理器
plt.rcParams["font.family"] = "Noto Sans CJK JP"  # 设置绘图默认字体
plt.rcParams["axes.unicode_minus"] = False  # 允许坐标轴正确显示负号


def loss_fn(w):
    """二维 Loss：L=(w1-2)^2+2*(w2+1)^2。"""
    w1, w2 = w  # 从参数向量中取出两个参数
    return (w1 - 2.0) ** 2 + 2.0 * (w2 + 1.0) ** 2  # 返回标量 Loss


def analytic_gradient(w):
    """解析梯度：把两个偏导数组成一个向量。"""
    w1, w2 = w  # 取出两个参数
    d_w1 = 2.0 * (w1 - 2.0)  # 计算 ∂L/∂w1
    d_w2 = 4.0 * (w2 + 1.0)  # 计算 ∂L/∂w2
    return np.array([d_w1, d_w2], dtype=float)  # 把两个偏导数组成梯度向量


def numerical_gradient(func, w, eps=1e-5):
    """用中心差分逐个参数估计梯度。"""
    grad = np.zeros_like(w, dtype=float)  # 创建与参数向量同 Shape 的梯度容器
    for i in range(w.size):  # 依次处理每一个参数
        w_plus = w.copy()  # 复制当前参数
        w_minus = w.copy()  # 再复制一份当前参数
        w_plus[i] += eps  # 只把第 i 个参数增加 eps
        w_minus[i] -= eps  # 只把第 i 个参数减少 eps
        grad[i] = (func(w_plus) - func(w_minus)) / (2.0 * eps)  # 中心差分估计该偏导
    return grad  # 返回完整梯度向量


print("=" * 76)  # 打印分隔线
print("实验1：把多个偏导数组成 Gradient")  # 输出实验标题
print("=" * 76)  # 打印分隔线
w0 = np.array([0.0, 1.0])  # 设定当前参数点 (w1,w2)=(0,1)
loss0 = loss_fn(w0)  # 计算当前 Loss
grad_analytic = analytic_gradient(w0)  # 计算解析梯度
grad_numeric = numerical_gradient(loss_fn, w0)  # 用数值差分估计梯度
print(f"当前参数 w={w0}")  # 打印当前参数
print(f"当前 Loss={loss0:.6f}")  # 打印当前损失
print(f"解析梯度 ∇L={grad_analytic}")  # 打印解析梯度
print(f"数值梯度 ∇L≈{grad_numeric}")  # 打印数值梯度
print(f"最大绝对误差={np.max(np.abs(grad_analytic - grad_numeric)):.10f}")  # 比较两种梯度
print(f"梯度 Shape={grad_analytic.shape}，参数 Shape={w0.shape}")  # 强调梯度和参数 Shape 一致
print(f"梯度范数 ||∇L||={np.linalg.norm(grad_analytic):.6f}")  # 计算梯度向量长度

print("\n" + "=" * 76)  # 换行并打印分隔线
print("实验2：梯度方向是否真的是局部上升最快方向？")  # 输出实验标题
print("=" * 76)  # 打印分隔线
step = 1e-3  # 设定非常小的移动距离，用于考察局部变化
angles = np.linspace(0.0, 2.0 * np.pi, 721)  # 生成大量方向角
changes = []  # 用列表保存每个方向上的 Loss 变化率
for theta in angles:  # 遍历所有方向
    direction = np.array([np.cos(theta), np.sin(theta)])  # 构造单位方向向量
    new_loss = loss_fn(w0 + step * direction)  # 沿该方向移动一个很小步长后计算 Loss
    changes.append((new_loss - loss0) / step)  # 用 ΔL/step 估计该方向上的变化率
changes = np.array(changes)  # 转成 NumPy 数组便于找最大值和最小值
max_idx = int(np.argmax(changes))  # 找到 Loss 增长最快的方向索引
min_idx = int(np.argmin(changes))  # 找到 Loss 下降最快的方向索引
max_direction = np.array([np.cos(angles[max_idx]), np.sin(angles[max_idx])])  # 得到最快上升方向
min_direction = np.array([np.cos(angles[min_idx]), np.sin(angles[min_idx])])  # 得到最快下降方向
grad_unit = grad_analytic / np.linalg.norm(grad_analytic)  # 把梯度归一化成单位向量
neg_grad_unit = -grad_unit  # 负梯度单位向量
print(f"单位梯度方向={grad_unit}")  # 打印理论最快上升方向
print(f"搜索到的最快上升方向≈{max_direction}")  # 打印数值搜索结果
print(f"单位负梯度方向={neg_grad_unit}")  # 打印理论最快下降方向
print(f"搜索到的最快下降方向≈{min_direction}")  # 打印数值搜索结果
print(f"梯度方向夹角余弦≈{np.dot(grad_unit, max_direction):.6f}")  # 越接近1说明方向越一致
print(f"负梯度方向夹角余弦≈{np.dot(neg_grad_unit, min_direction):.6f}")  # 越接近1说明方向越一致

print("\n" + "=" * 76)  # 换行并打印分隔线
print("实验3：沿负梯度走一步，Loss 一定下降吗？")  # 输出实验标题
print("=" * 76)  # 打印分隔线
for learning_rate in [0.01, 0.10, 0.25, 0.50, 1.00]:  # 测试从很小到很大的多个步长
    w_new = w0 - learning_rate * grad_analytic  # 按负梯度方向更新参数
    new_loss = loss_fn(w_new)  # 计算更新后的 Loss
    print(
        f"lr={learning_rate:>4.2f} -> 新参数={w_new}, "
        f"新 Loss={new_loss:>8.4f}, ΔL={new_loss - loss0:>8.4f}"
    )  # 输出每个步长的结果
print("结论：负梯度给出局部下降方向，但步长太大仍然可能跨过低谷，让 Loss 反而变大。")  # 强调梯度和学习率的区别

print("\n" + "=" * 76)  # 换行并打印分隔线
print("实验4：连续沿负梯度更新时，Loss 与梯度范数怎样变化？")  # 输出实验标题
print("=" * 76)  # 打印分隔线
w = w0.copy()  # 从初始点开始
lr = 0.15  # 选择一个能够稳定下降的学习率
history = []  # 保存每一步的参数、Loss 和梯度范数
for step_id in range(12):  # 连续做12次更新
    g = analytic_gradient(w)  # 计算当前位置的梯度
    current_loss = loss_fn(w)  # 计算当前位置的 Loss
    grad_norm = np.linalg.norm(g)  # 计算梯度范数
    history.append((step_id, w.copy(), current_loss, grad_norm))  # 保存当前状态
    w = w - lr * g  # 沿负梯度方向更新参数
for step_id, params, current_loss, grad_norm in history:  # 输出所有记录
    print(
        f"step={step_id:02d}  w={params}  "
        f"Loss={current_loss:9.6f}  ||∇L||={grad_norm:9.6f}"
    )  # 打印每一步的参数、损失和梯度长度

print("\n" + "=" * 76)  # 换行并打印分隔线
print("实验5：矩阵参数的 Gradient 为什么与参数 Shape 相同？")  # 输出实验标题
print("=" * 76)  # 打印分隔线
X = np.array([[1.0, 2.0], [0.0, -1.0], [2.0, 1.0]])  # 三个样本，每个样本2个输入特征
W = np.array([[0.5, -0.2], [1.0, 0.3]])  # 2x2 权重矩阵
Y = np.array([[2.0, 0.0], [-1.0, -0.5], [2.5, 0.0]])  # 每个样本有2个目标输出
P = X @ W  # 计算模型预测，Shape=(3,2)
E = P - Y  # 计算误差矩阵，Shape=(3,2)
mse = np.mean(E**2)  # 对全部6个输出元素求 MSE
grad_W = (2.0 / E.size) * (X.T @ E)  # 根据矩阵微分结果得到对 W 的梯度
print(f"X.shape={X.shape}")  # 打印输入 Shape
print(f"W.shape={W.shape}")  # 打印参数 Shape
print(f"Prediction.shape={P.shape}")  # 打印预测 Shape
print(f"Loss={mse:.6f}")  # 打印当前 MSE
print(f"grad_W.shape={grad_W.shape}")  # 梯度 Shape 应与 W 一致
print("grad_W=")  # 打印梯度矩阵标题
print(grad_W)  # 打印每一个权重元素对应的偏导数

# ---------------- 绘图1：Loss 等高线 + 梯度与负梯度箭头 ----------------
w1_vals = np.linspace(-1.0, 4.0, 240)  # 生成横轴参数网格
w2_vals = np.linspace(-3.0, 2.5, 240)  # 生成纵轴参数网格
W1, W2 = np.meshgrid(w1_vals, w2_vals)  # 构造二维网格
Z = (W1 - 2.0) ** 2 + 2.0 * (W2 + 1.0) ** 2  # 计算网格上每个点的 Loss
plt.figure(figsize=(8, 6))  # 创建图像
levels = np.geomspace(0.05, float(Z.max()), 18)  # 使用对数间距的等高线层级
plt.contour(W1, W2, Z, levels=levels)  # 绘制 Loss 等高线
plt.scatter([w0[0]], [w0[1]], s=60, label="当前参数 (0,1)")  # 标出当前参数点
plt.scatter([2.0], [-1.0], marker="x", s=80, label="最低点 (2,-1)")  # 标出最低点
arrow_scale = 0.12  # 为可视化设置箭头缩放比例
plt.arrow(w0[0], w0[1], arrow_scale * grad_analytic[0], arrow_scale * grad_analytic[1], width=0.015, length_includes_head=True, label="梯度方向")  # 画梯度箭头
plt.arrow(w0[0], w0[1], -arrow_scale * grad_analytic[0], -arrow_scale * grad_analytic[1], width=0.015, length_includes_head=True, label="负梯度方向")  # 画负梯度箭头
plt.xlabel("w1")  # 设置横轴标签
plt.ylabel("w2")  # 设置纵轴标签
plt.title("Gradient：梯度指向局部上升最快，负梯度指向局部下降最快")  # 设置标题
plt.legend()  # 显示图例
plt.tight_layout()  # 自动调整边距
plt.savefig("/mnt/data/deep_learning_lesson20/final/figures/lesson20_gradient_direction.png", dpi=180)  # 保存图片
plt.close()  # 关闭图像释放内存

# ---------------- 绘图2：不同学习率沿负梯度一步 ----------------
learning_rates = np.array([0.0, 0.01, 0.10, 0.25, 0.50, 1.00])  # 定义学习率数组
one_step_losses = []  # 保存每个学习率的一步更新后 Loss
for eta in learning_rates:  # 遍历学习率
    one_step_losses.append(loss_fn(w0 - eta * grad_analytic))  # 计算一步更新后的 Loss
plt.figure(figsize=(8, 5))  # 创建图像
plt.plot(learning_rates, one_step_losses, marker="o")  # 绘制学习率与一步 Loss 的关系
plt.axhline(loss0, linestyle="--", linewidth=1.2, label="更新前 Loss")  # 画出更新前 Loss 基准线
plt.xlabel("学习率 / 步长 η")  # 设置横轴标签
plt.ylabel("一步更新后的 Loss")  # 设置纵轴标签
plt.title("负梯度给方向，但步长过大仍可能让 Loss 上升")  # 设置标题
plt.legend()  # 显示图例
plt.tight_layout()  # 自动调整边距
plt.savefig("/mnt/data/deep_learning_lesson20/final/figures/lesson20_learning_rate_one_step.png", dpi=180)  # 保存图片
plt.close()  # 关闭图像释放内存

# ---------------- 绘图3：连续更新的 Loss 与梯度范数 ----------------
steps = np.array([item[0] for item in history])  # 提取 step 编号
loss_history = np.array([item[2] for item in history])  # 提取 Loss 历史
grad_norm_history = np.array([item[3] for item in history])  # 提取梯度范数历史
plt.figure(figsize=(8, 5))  # 创建 Loss 曲线图
plt.plot(steps, loss_history, marker="o")  # 绘制 Loss 随 step 变化
plt.xlabel("更新步数")  # 设置横轴标签
plt.ylabel("Loss")  # 设置纵轴标签
plt.title("沿负梯度连续更新：Loss 逐步下降")  # 设置标题
plt.tight_layout()  # 自动调整边距
plt.savefig("/mnt/data/deep_learning_lesson20/final/figures/lesson20_loss_descent.png", dpi=180)  # 保存图片
plt.close()  # 关闭图像释放内存

plt.figure(figsize=(8, 5))  # 创建梯度范数曲线图
plt.plot(steps, grad_norm_history, marker="o")  # 绘制梯度范数随 step 变化
plt.xlabel("更新步数")  # 设置横轴标签
plt.ylabel("||∇L||")  # 设置纵轴标签
plt.title("接近最低点时，这个例子中的梯度范数逐步变小")  # 设置标题
plt.tight_layout()  # 自动调整边距
plt.savefig("/mnt/data/deep_learning_lesson20/final/figures/lesson20_gradient_norm.png", dpi=180)  # 保存图片
plt.close()  # 关闭图像释放内存

print("\n已生成 4 张实验图：")  # 输出图片列表标题
print("1. lesson20_gradient_direction.png")  # 第一张图片文件名
print("2. lesson20_learning_rate_one_step.png")  # 第二张图片文件名
print("3. lesson20_loss_descent.png")  # 第三张图片文件名
print("4. lesson20_gradient_norm.png")  # 第四张图片文件名
