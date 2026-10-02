# 第29讲实验：PyTorch Autograd —— loss.backward() 到底怎样自动得到梯度
# 运行方式：python lesson29_autograd_experiment.py

from pathlib import Path  # 用于构造稳定的输出目录
import sys  # 用于打印 Python 版本
import numpy as np  # 用于数值记录与绘图辅助
import torch  # PyTorch 核心库
from torch import nn  # 神经网络模块
import torch.nn.functional as F  # 常用函数式接口
import matplotlib.pyplot as plt  # 绘图
from matplotlib import font_manager  # 中文字体支持

# ------------------------------
# 中文字体与输出目录
# ------------------------------
CJK_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"  # 容器中的中文字体
font_manager.fontManager.addfont(CJK_FONT_PATH)  # 注册中文字体
CJK_FONT = font_manager.FontProperties(fname=CJK_FONT_PATH)  # 构造字体对象
plt.rcParams["axes.unicode_minus"] = False  # 避免负号显示异常

BASE_DIR = Path(__file__).resolve().parent  # 当前脚本所在目录
FIG_DIR = BASE_DIR / "figures"  # 图像输出目录
FIG_DIR.mkdir(exist_ok=True)  # 如果目录不存在则创建


def title(text):
    """打印实验分隔标题。"""
    print("\n" + "=" * 82)  # 打印分隔线
    print(text)  # 打印标题
    print("=" * 82)  # 打印分隔线


def max_abs_diff(a, b):
    """返回两个 Tensor 之间的最大绝对误差。"""
    return torch.max(torch.abs(a - b)).item()  # 计算并转成 Python 浮点数


# -----------------------------------------------------------------------------
# 实验0：环境信息与随机种子
# -----------------------------------------------------------------------------
title("实验0：环境信息与随机种子")
print("Python:", sys.version.split()[0])  # 打印 Python 版本
print("NumPy:", np.__version__)  # 打印 NumPy 版本
print("PyTorch:", torch.__version__)  # 打印 PyTorch 版本
print("CUDA available:", torch.cuda.is_available())  # 检查 CUDA 是否可用

torch.manual_seed(29)  # 固定 PyTorch 随机种子
np.random.seed(29)  # 固定 NumPy 随机种子

# -----------------------------------------------------------------------------
# 实验1：requires_grad、is_leaf、grad_fn —— 计算图是怎样开始记录的？
# -----------------------------------------------------------------------------
title("实验1：requires_grad、is_leaf、grad_fn 与动态计算图")

x = torch.tensor(3.0, requires_grad=True)  # 创建一个需要梯度的叶子 Tensor
y = x * x  # y = x^2；PyTorch 会记录乘法节点
z = y + 2.0  # z = y + 2；继续记录加法节点

print("x =", x)  # 打印 x
print("x.requires_grad =", x.requires_grad)  # True：要求追踪梯度
print("x.is_leaf =", x.is_leaf)  # True：x 是用户创建的叶子 Tensor
print("x.grad_fn =", x.grad_fn)  # 叶子 Tensor 通常没有 grad_fn
print("y =", y)  # 打印中间变量 y
print("y.requires_grad =", y.requires_grad)  # True：由需要梯度的 x 计算而来
print("y.is_leaf =", y.is_leaf)  # False：y 是中间结果
print("y.grad_fn =", type(y.grad_fn).__name__)  # 查看对应的反向节点类型
print("z =", z)  # 打印中间变量 z
print("z.grad_fn =", type(z.grad_fn).__name__)  # 查看加法对应的反向节点
print("在 backward 之前，x.grad =", x.grad)  # 尚未反向传播，因此梯度为空

# -----------------------------------------------------------------------------
# 实验2：最小 backward —— x^2 在 x=3 处的梯度应该是6
# -----------------------------------------------------------------------------
title("实验2：最小 backward —— x^2 在 x=3 处自动得到梯度6")

x2 = torch.tensor(3.0, requires_grad=True)  # 新建一个叶子 Tensor，避免复用旧计算图
loss2 = x2 ** 2  # 构造标量 Loss：L=x^2
print("loss =", loss2.item())  # Forward 得到 L=9
print("before backward: x.grad =", x2.grad)  # backward 前还没有梯度
loss2.backward()  # 从标量 Loss 出发执行反向传播
print("after backward:  x.grad =", x2.grad.item())  # 应得到 dL/dx=2x=6
print("解析结果 2*x =", 2 * x2.item())  # 打印解析结果用于对照

# 绘图：函数曲线与 x=3 的切线
xs = np.linspace(-0.5, 5.0, 300)  # 构造横轴采样点
ys = xs ** 2  # 计算 y=x^2
x0 = 3.0  # 当前点
slope = 6.0  # x=3 处导数
line = x0 ** 2 + slope * (xs - x0)  # 当前点切线
plt.figure(figsize=(8.2, 5.0))  # 创建图像
plt.plot(xs, ys, label=r"$L=x^2$")  # 绘制 Loss 曲线
plt.plot(xs, line, "--", label="x=3 处切线，斜率=6")  # 绘制切线
plt.scatter([x0], [x0 ** 2], s=55)  # 标出当前点
plt.xlabel("参数 x", fontproperties=CJK_FONT)  # 横轴
plt.ylabel("Loss", fontproperties=CJK_FONT)  # 纵轴
plt.title("Autograd 得到的梯度 = 当前点的局部斜率", fontproperties=CJK_FONT)  # 标题
plt.legend(prop=CJK_FONT)  # 图例
plt.tight_layout()  # 自动调整布局
plt.savefig(FIG_DIR / "lesson29_scalar_autograd.png", dpi=180)  # 保存图片
plt.close()  # 关闭图像

# -----------------------------------------------------------------------------
# 实验3：把第24讲的单神经元手工梯度交给 Autograd 再算一次
# -----------------------------------------------------------------------------
title("实验3：单神经元 —— 手工 Backprop 与 Autograd 三个梯度完全对齐")

x3 = torch.tensor([2.0, -1.0, 3.0], requires_grad=True)  # 输入也要求梯度，便于检查 dL/dx
w3 = torch.tensor([0.5, -2.0, 1.0], requires_grad=True)  # 可训练 Weight
b3 = torch.tensor(0.2, requires_grad=True)  # 可训练 Bias
target3 = torch.tensor(5.0)  # 监督目标不需要梯度

z3 = torch.dot(x3, w3) + b3  # z = x·w + b
a3 = F.relu(z3)  # a = ReLU(z)
loss3 = (a3 - target3) ** 2  # L = (a-y)^2

print("Forward: z =", z3.item())  # 预期 6.2
print("Forward: a =", a3.item())  # ReLU 后仍为 6.2
print("Forward: loss =", loss3.item())  # 预期 1.44
print("before backward: w.grad =", w3.grad)  # 反向前为空

loss3.backward()  # PyTorch 自动沿计算图执行 Chain Rule

manual_dw = torch.tensor([4.8, -2.4, 7.2])  # 第24讲已经手工得到的 dL/dw
manual_db = torch.tensor(2.4)  # 第24讲手工 dL/db
manual_dx = torch.tensor([1.2, -4.8, 2.4])  # 第24讲手工 dL/dx

print("Autograd dL/dw =", w3.grad)  # 打印自动梯度
print("Manual   dL/dw =", manual_dw)  # 打印手工梯度
print("Autograd dL/db =", b3.grad.item())  # 打印 Bias 梯度
print("Manual   dL/db =", manual_db.item())  # 打印手工 Bias 梯度
print("Autograd dL/dx =", x3.grad)  # 打印输入梯度
print("Manual   dL/dx =", manual_dx)  # 打印手工输入梯度
print("dw max abs diff =", max_abs_diff(w3.grad, manual_dw))  # 对比 dL/dw
print("db abs diff =", abs(b3.grad.item() - manual_db.item()))  # 对比 dL/db
print("dx max abs diff =", max_abs_diff(x3.grad, manual_dx))  # 对比 dL/dx

# 绘制手工梯度与 Autograd 梯度对照
labels = ["dw1", "dw2", "dw3", "db", "dx1", "dx2", "dx3"]  # 梯度名称
auto_values = np.array([
    w3.grad[0].item(), w3.grad[1].item(), w3.grad[2].item(), b3.grad.item(),
    x3.grad[0].item(), x3.grad[1].item(), x3.grad[2].item(),
])  # Autograd 梯度
manual_values = np.array([4.8, -2.4, 7.2, 2.4, 1.2, -4.8, 2.4])  # 手工梯度
positions = np.arange(len(labels))  # 横轴位置
width = 0.36  # 柱宽
plt.figure(figsize=(9.0, 5.0))  # 创建图像
plt.bar(positions - width / 2, manual_values, width=width, label="手工 Backprop")  # 手工结果
plt.bar(positions + width / 2, auto_values, width=width, label="PyTorch Autograd")  # 自动结果
plt.axhline(0.0, linewidth=0.8)  # 零线
plt.xticks(positions, labels)  # 横轴标签
plt.ylabel("Gradient", fontproperties=CJK_FONT)  # 纵轴
plt.title("手工 Backprop 与 Autograd 梯度逐项一致", fontproperties=CJK_FONT)  # 标题
plt.legend(prop=CJK_FONT)  # 图例
plt.tight_layout()  # 调整布局
plt.savefig(FIG_DIR / "lesson29_manual_vs_autograd.png", dpi=180)  # 保存图片
plt.close()  # 关闭图像

# -----------------------------------------------------------------------------
# 实验4：梯度为什么会累加？为什么训练循环要 zero_grad？
# -----------------------------------------------------------------------------
title("实验4：Gradient Accumulation —— backward 默认把新梯度加到 .grad 中")

p = torch.tensor(2.0, requires_grad=True)  # 建立一个简单参数 p
loss_a = p ** 2  # 第一次 Forward：L=p^2
loss_a.backward()  # 第一次 backward，梯度应该为4
grad_after_1 = p.grad.item()  # 记录第一次梯度
print("第一次 backward 后 p.grad =", grad_after_1)  # 输出4

loss_b = p ** 2  # 重新 Forward，得到一张新的动态计算图
loss_b.backward()  # 第二次 backward；默认不是覆盖，而是继续累加
grad_after_2 = p.grad.item()  # 记录累加后的梯度
print("第二次 backward 后 p.grad =", grad_after_2)  # 输出8

p.grad.zero_()  # 手工把梯度清零
print("zero_() 后 p.grad =", p.grad.item())  # 输出0

loss_c = p ** 2  # 再重新 Forward
loss_c.backward()  # 再反向一次
grad_after_zero = p.grad.item()  # 记录清零后的新梯度
print("清零后再 backward，p.grad =", grad_after_zero)  # 应重新得到4

# 绘图：梯度累加现象
names = ["1次 backward", "2次 backward\n未清零", "zero 后\n再 backward"]  # 三种状态
values = [grad_after_1, grad_after_2, grad_after_zero]  # 对应梯度数值
plt.figure(figsize=(7.4, 4.8))  # 创建图像
plt.bar(np.arange(3), values)  # 绘制柱状图
plt.xticks(np.arange(3), names, fontproperties=CJK_FONT)  # 横轴标签
plt.ylabel("p.grad", fontproperties=CJK_FONT)  # 纵轴
plt.title("PyTorch 的 .grad 默认采用累加语义", fontproperties=CJK_FONT)  # 标题
plt.tight_layout()  # 调整布局
plt.savefig(FIG_DIR / "lesson29_gradient_accumulation.png", dpi=180)  # 保存图片
plt.close()  # 关闭图像

# -----------------------------------------------------------------------------
# 实验5：为什么非标量 Tensor 不能直接 backward？
# -----------------------------------------------------------------------------
title("实验5：非标量输出 —— backward 需要先定义怎样把多个输出聚合成一个目标")

v = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)  # 三维输入向量
y_vec = v ** 2  # 得到三维输出，而不是一个标量
print("y_vec =", y_vec)  # [1,4,9]
print("y_vec.shape =", tuple(y_vec.shape))  # Shape=(3,)

try:
    y_vec.backward()  # 故意直接对非标量调用 backward
except RuntimeError as exc:
    print("直接 y_vec.backward() 的报错类型:", type(exc).__name__)  # 打印报错类型
    print("报错说明:", str(exc).split("\n")[0])  # 只打印第一行，避免输出过长

loss_vec = y_vec.sum()  # 明确把多个输出聚合成一个标量 Loss
loss_vec.backward()  # 现在可以从标量出发反向传播
print("对 y_vec.sum() backward 后 v.grad =", v.grad)  # 应为 [2,4,6]

# 再展示更一般的 vector-Jacobian product：给每个输出指定上游梯度。
v2 = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)  # 新建变量以获得新计算图
y_vec2 = v2 ** 2  # 同样得到向量输出
upstream = torch.tensor([1.0, 0.5, 0.0])  # 人为指定来自后续的上游梯度
custom_grad = torch.autograd.grad(y_vec2, v2, grad_outputs=upstream)[0]  # 计算加权后的梯度
print("grad_outputs =", upstream)  # 打印上游梯度
print("torch.autograd.grad 结果 =", custom_grad)  # 应为 [2,2,0]

# -----------------------------------------------------------------------------
# 实验6：no_grad 与 detach —— 怎样主动切断梯度记录？
# -----------------------------------------------------------------------------
title("实验6：torch.no_grad() 与 detach() —— 主动停止记录梯度")

q = torch.tensor(2.0, requires_grad=True)  # 创建需要梯度的变量
r = q * 3.0  # 正常计算，记录计算图
r_detached = r.detach()  # detach 得到共享数据但不继续追踪梯度的 Tensor

with torch.no_grad():  # 临时关闭梯度记录
    s = q * 4.0  # 这个计算不会被放进 Autograd 图

print("q.requires_grad =", q.requires_grad)  # True
print("r.requires_grad =", r.requires_grad)  # True
print("r.grad_fn =", type(r.grad_fn).__name__)  # 有 grad_fn
print("r_detached.requires_grad =", r_detached.requires_grad)  # False
print("r_detached.grad_fn =", r_detached.grad_fn)  # None
print("s.requires_grad =", s.requires_grad)  # False
print("s.grad_fn =", s.grad_fn)  # None

# -----------------------------------------------------------------------------
# 实验7：nn.Linear + MSE —— 参数梯度 Shape 与手工矩阵公式对照
# -----------------------------------------------------------------------------
title("实验7：nn.Linear + MSE —— 参数梯度 Shape 与手工矩阵公式对齐")

linear7 = nn.Linear(3, 2, bias=True)  # 创建一个3->2的线性层
known_w7 = torch.tensor(
    [[0.5, -1.0, 2.0],
     [1.0,  0.5, -0.5]],
    dtype=torch.float32,
)  # 固定 Weight
known_b7 = torch.tensor([0.2, -0.3], dtype=torch.float32)  # 固定 Bias

with torch.no_grad():  # 参数赋值本身不记录梯度
    linear7.weight.copy_(known_w7)  # 写入 Weight
    linear7.bias.copy_(known_b7)  # 写入 Bias

X7 = torch.tensor(
    [[1.0, 2.0, -1.0],
     [0.0, 1.0,  2.0]],
    dtype=torch.float32,
    requires_grad=True,
)  # 两个样本，每个3个特征，并追踪输入梯度
T7 = torch.tensor(
    [[-3.0, 2.0],
     [ 3.0, 0.0]],
    dtype=torch.float32,
)  # 回归目标

Y7 = linear7(X7)  # Forward：Y = X W^T + b
loss7 = F.mse_loss(Y7, T7, reduction="mean")  # 对全部4个输出元素求平均平方误差
print("Y.shape =", tuple(Y7.shape))  # (2,2)
print("loss =", loss7.item())  # 打印 Loss
print("before backward: weight.grad =", linear7.weight.grad)  # 反向前梯度为空

loss7.backward()  # 自动计算参数梯度与输入梯度

print("weight.grad.shape =", tuple(linear7.weight.grad.shape))  # 应与 weight 相同：(2,3)
print("bias.grad.shape =", tuple(linear7.bias.grad.shape))  # 应与 bias 相同：(2,)
print("X.grad.shape =", tuple(X7.grad.shape))  # 应与输入相同：(2,3)
print("Autograd weight.grad =\n", linear7.weight.grad)  # 打印自动 dL/dW
print("Autograd bias.grad =", linear7.bias.grad)  # 打印自动 dL/db
print("Autograd X.grad =\n", X7.grad)  # 打印自动 dL/dX

# 手工推导 MSE 的 dL/dY：mean over 4 elements，所以除以 numel。
dY7 = 2.0 * (Y7.detach() - T7) / Y7.numel()  # dL/dY
manual_dW7 = dY7.T @ X7.detach()  # 对 PyTorch Weight 存储形式，dW = dY^T @ X
manual_db7 = dY7.sum(dim=0)  # Bias 对 Batch 维求和
manual_dX7 = dY7 @ linear7.weight.detach()  # dX = dY @ W

print("Manual weight.grad =\n", manual_dW7)  # 打印手工 dW
print("Manual bias.grad =", manual_db7)  # 打印手工 db
print("Manual X.grad =\n", manual_dX7)  # 打印手工 dX
print("weight grad max diff =", max_abs_diff(linear7.weight.grad, manual_dW7))  # 对比 dW
print("bias grad max diff =", max_abs_diff(linear7.bias.grad, manual_db7))  # 对比 db
print("X grad max diff =", max_abs_diff(X7.grad, manual_dX7))  # 对比 dX

# -----------------------------------------------------------------------------
# 实验8：一次手工参数更新 —— backward 只算梯度，不会自动修改参数
# -----------------------------------------------------------------------------
title("实验8：backward 只计算梯度；参数是否更新是另一件事")

# 保存更新前的 Loss。
loss_before = loss7.item()  # 记录原 Loss
lr = 0.05  # 设置一个较小学习率

with torch.no_grad():  # 参数更新本身不应该进入 Autograd 图
    linear7.weight -= lr * linear7.weight.grad  # 按负梯度方向更新 Weight
    linear7.bias -= lr * linear7.bias.grad  # 按负梯度方向更新 Bias

# 清空旧梯度，避免下一轮与旧梯度累加。
linear7.zero_grad()  # 将参数梯度清零/置空（版本实现可能不同）
X7.grad = None  # 手工清空输入梯度，便于观察下一轮

# 重新 Forward，获得新图和新 Loss。
Y7_after = linear7(X7)  # 使用更新后的参数重新预测
loss7_after = F.mse_loss(Y7_after, T7, reduction="mean")  # 重新计算 Loss
print("loss before update =", loss_before)  # 更新前 Loss
print("loss after update  =", loss7_after.item())  # 更新后 Loss
print("weight.grad after zero_grad =", linear7.weight.grad)  # 应为空或零，取决于实现
print("bias.grad after zero_grad =", linear7.bias.grad)  # 应为空或零，取决于实现

# -----------------------------------------------------------------------------
# 总结
# -----------------------------------------------------------------------------
title("实验总结")
print("1. requires_grad=True 让相关运算开始被 Autograd 记录。")
print("2. grad_fn 描述非叶子 Tensor 是由什么运算产生的。")
print("3. backward() 从标量目标反向遍历计算图，并把结果写入叶子 Tensor 的 .grad。")
print("4. .grad 默认累加，因此训练循环需要主动 zero_grad / set_to_none。")
print("5. backward 只负责计算梯度，不会自动更新参数。")
print("6. no_grad / detach 可以主动切断梯度记录，但语义和使用场景需要区分。")
print("7. Autograd 的数学本质仍然是前面学过的 Computational Graph + Chain Rule + 梯度累加。")
print("\n实验完成。图像已写入:", FIG_DIR)
