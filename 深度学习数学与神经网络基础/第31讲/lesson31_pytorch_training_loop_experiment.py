# 第31讲实验：PyTorch Training Loop
# 目标：逐步观察 zero_grad -> forward -> loss -> backward -> step 每一步到底做什么。

from pathlib import Path  # 处理输出路径
import sys  # 查看 Python 版本
import numpy as np  # 数值辅助
import torch  # PyTorch 主库
import torch.nn as nn  # 神经网络模块
from matplotlib import pyplot as plt  # 绘图
from matplotlib import font_manager  # 中文字体

# -----------------------------------------------------------------------------
# 中文绘图字体
# -----------------------------------------------------------------------------
CJK_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"  # 中文字体路径
font_manager.fontManager.addfont(CJK_FONT_PATH)  # 注册字体
CJK_FONT = font_manager.FontProperties(fname=CJK_FONT_PATH)  # 创建中文字体对象
plt.rcParams["axes.unicode_minus"] = False  # 避免负号显示异常

BASE_DIR = Path(__file__).resolve().parent  # 当前脚本目录
FIG_DIR = BASE_DIR / "figures"  # 图像目录
FIG_DIR.mkdir(exist_ok=True)  # 确保目录存在


def title(text):
    """打印实验分隔标题。"""
    print("\n" + "=" * 92)  # 分隔线
    print(text)  # 标题正文
    print("=" * 92)  # 分隔线


def make_dataset():
    """创建固定的小型回归数据：y = 2x + 1。"""
    x = torch.linspace(-2.0, 2.0, steps=41).reshape(-1, 1)  # 41 个输入点，Shape=(41,1)
    y = 2.0 * x + 1.0  # 对应真实目标，Shape=(41,1)
    return x, y  # 返回输入与目标


def make_model(seed=31):
    """创建一个最简单的 Linear(1,1) 模型。"""
    torch.manual_seed(seed)  # 固定初始化，便于公平对照
    return nn.Linear(1, 1)  # 一个 Weight + 一个 Bias


# -----------------------------------------------------------------------------
# 实验0：环境信息
# -----------------------------------------------------------------------------
title("实验0：环境信息")
print("Python:", sys.version.split()[0])  # Python 版本
print("PyTorch:", torch.__version__)  # PyTorch 版本
print("CUDA available:", torch.cuda.is_available())  # 是否可用 CUDA

X, Y = make_dataset()  # 创建固定数据
print("X.shape =", tuple(X.shape))  # 输入 Shape
print("Y.shape =", tuple(Y.shape))  # 目标 Shape
print("数据规则：Y = 2 * X + 1")  # 真实规律


# -----------------------------------------------------------------------------
# 实验1：把“一次训练 Step”完全拆开
# -----------------------------------------------------------------------------
title("实验1：一次训练 Step —— zero_grad -> forward -> loss -> backward -> step")

model = make_model(seed=31)  # 固定初始化的模型
criterion = nn.MSELoss()  # MSE Loss
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)  # SGD 优化器，学习率 0.05

weight_before = model.weight.detach().clone()  # 记录 step 前 Weight
bias_before = model.bias.detach().clone()  # 记录 step 前 Bias
print("初始 weight =", model.weight.detach().item())  # 初始 Weight
print("初始 bias   =", model.bias.detach().item())  # 初始 Bias
print("zero_grad 前 weight.grad =", model.weight.grad)  # 第一次训练前通常是 None

optimizer.zero_grad()  # 1) 清理旧梯度，防止和本轮梯度累加
print("zero_grad 后 weight.grad =", model.weight.grad)  # set_to_none 默认下仍可为 None

pred = model(X)  # 2) Forward：根据当前参数得到预测
loss = criterion(pred, Y)  # 3) Loss：比较 Prediction 与 Target
print("forward 后 pred.shape =", tuple(pred.shape))  # 预测 Shape
print("当前 loss =", loss.item())  # 当前 Loss
print("backward 前 weight.grad =", model.weight.grad)  # 此时还没算梯度

loss.backward()  # 4) Backward：Autograd 沿计算图计算梯度
print("backward 后 dL/dweight =", model.weight.grad.item())  # Weight 梯度
print("backward 后 dL/dbias   =", model.bias.grad.item())  # Bias 梯度

weight_after_backward = model.weight.detach().clone()  # backward 后参数快照
bias_after_backward = model.bias.detach().clone()  # backward 后参数快照
print("backward 是否修改 weight:", not torch.equal(weight_before, weight_after_backward))  # 应为 False
print("backward 是否修改 bias  :", not torch.equal(bias_before, bias_after_backward))  # 应为 False

optimizer.step()  # 5) Step：优化器根据 .grad 真正修改 Parameter
weight_after_step = model.weight.detach().clone()  # step 后 Weight
bias_after_step = model.bias.detach().clone()  # step 后 Bias
print("step 后 weight =", weight_after_step.item())  # 更新后的 Weight
print("step 后 bias   =", bias_after_step.item())  # 更新后的 Bias
print("step 是否修改 weight:", not torch.equal(weight_after_backward, weight_after_step))  # 应为 True
print("step 是否修改 bias  :", not torch.equal(bias_after_backward, bias_after_step))  # 应为 True

with torch.no_grad():  # 评估时不需要构建梯度图
    loss_after_one_step = criterion(model(X), Y).item()  # 再算一次 Loss
print("一步更新后 loss =", loss_after_one_step)  # 通常下降


# -----------------------------------------------------------------------------
# 实验2：标准 Training Loop —— 让模型学会 y = 2x + 1
# -----------------------------------------------------------------------------
title("实验2：标准 PyTorch Training Loop")

train_model = make_model(seed=31)  # 与实验1相同初始化
train_optimizer = torch.optim.SGD(train_model.parameters(), lr=0.05)  # SGD
train_criterion = nn.MSELoss()  # MSE

loss_history = []  # 记录每个 Step 的 Loss
weight_history = []  # 记录 Weight
bias_history = []  # 记录 Bias
steps = 120  # 训练 120 个 Step

for step in range(steps):  # 开始训练循环
    train_optimizer.zero_grad()  # A. 清理上一轮 Gradient
    prediction = train_model(X)  # B. Forward
    train_loss = train_criterion(prediction, Y)  # C. 计算 Loss
    train_loss.backward()  # D. Backward，计算所有 Parameter 的 Gradient
    train_optimizer.step()  # E. 根据 Gradient 更新 Parameter

    loss_history.append(train_loss.item())  # 保存当前 Loss
    weight_history.append(train_model.weight.detach().item())  # 保存当前 Weight
    bias_history.append(train_model.bias.detach().item())  # 保存当前 Bias

    if step in [0, 1, 2, 4, 9, 29, 59, 119]:  # 只打印若干关键 Step
        print(
            f"step={step:3d} "
            f"loss={train_loss.item():.8f} "
            f"weight={train_model.weight.detach().item():.6f} "
            f"bias={train_model.bias.detach().item():.6f}"
        )  # 输出训练轨迹

print("最终 weight =", train_model.weight.detach().item())  # 应接近 2
print("最终 bias   =", train_model.bias.detach().item())  # 应接近 1
print("最终 loss   =", loss_history[-1])  # 应非常小

# 训练 Loss 曲线
plt.figure(figsize=(7.2, 4.6))  # 创建画布
plt.plot(np.arange(steps), loss_history)  # 绘制 Loss
plt.xlabel("Step", fontproperties=CJK_FONT)  # 横轴
plt.ylabel("MSE Loss", fontproperties=CJK_FONT)  # 纵轴
plt.title("标准 Training Loop：Loss 随训练下降", fontproperties=CJK_FONT)  # 标题
plt.tight_layout()  # 自动排版
plt.savefig(FIG_DIR / "lesson31_training_loss.png", dpi=180)  # 保存图片
plt.close()  # 关闭画布

# Weight / Bias 收敛轨迹
plt.figure(figsize=(7.2, 4.6))  # 创建画布
plt.plot(np.arange(steps), weight_history, label="Weight")  # Weight 轨迹
plt.plot(np.arange(steps), bias_history, label="Bias")  # Bias 轨迹
plt.axhline(2.0, linestyle="--", label="真实 Weight=2")  # 真实 Weight
plt.axhline(1.0, linestyle="--", label="真实 Bias=1")  # 真实 Bias
plt.xlabel("Step", fontproperties=CJK_FONT)  # 横轴
plt.ylabel("参数值", fontproperties=CJK_FONT)  # 纵轴
plt.title("Parameter 在 Training Loop 中逐步接近真实值", fontproperties=CJK_FONT)  # 标题
plt.legend(prop=CJK_FONT)  # 图例
plt.tight_layout()  # 自动排版
plt.savefig(FIG_DIR / "lesson31_parameter_trajectory.png", dpi=180)  # 保存图片
plt.close()  # 关闭画布


# -----------------------------------------------------------------------------
# 实验3：故意省略 zero_grad —— Gradient 为什么会累加？
# -----------------------------------------------------------------------------
title("实验3：故意省略 zero_grad —— Gradient Accumulation")

acc_model = make_model(seed=31)  # 相同初始化
acc_criterion = nn.MSELoss()  # MSE

pred1 = acc_model(X)  # 第一次 Forward
loss1 = acc_criterion(pred1, Y)  # 第一次 Loss
loss1.backward()  # 第一次 Backward
first_weight_grad = acc_model.weight.grad.detach().item()  # 第一次 Weight Gradient
first_bias_grad = acc_model.bias.grad.detach().item()  # 第一次 Bias Gradient
print("第一次 backward 后 weight.grad =", first_weight_grad)  # 第一次梯度
print("第一次 backward 后 bias.grad   =", first_bias_grad)  # 第一次梯度

pred2 = acc_model(X)  # 参数未修改，再做相同 Forward
loss2 = acc_criterion(pred2, Y)  # 相同 Loss
loss2.backward()  # 没有清零，第二次梯度会累加到 .grad
second_weight_grad = acc_model.weight.grad.detach().item()  # 累加后的 Weight Gradient
second_bias_grad = acc_model.bias.grad.detach().item()  # 累加后的 Bias Gradient
print("未清零，第二次 backward 后 weight.grad =", second_weight_grad)  # 约为两倍
print("未清零，第二次 backward 后 bias.grad   =", second_bias_grad)  # 约为两倍
print("weight.grad 比例 =", second_weight_grad / first_weight_grad)  # 应接近 2
print("bias.grad 比例   =", second_bias_grad / first_bias_grad)  # 应接近 2

acc_model.zero_grad(set_to_none=True)  # 清理梯度
print("model.zero_grad(set_to_none=True) 后 weight.grad =", acc_model.weight.grad)  # None

# 梯度累加图
plt.figure(figsize=(7.2, 4.6))  # 创建画布
positions = np.arange(2)  # 两个参数位置
width = 0.36  # 柱宽
plt.bar(positions - width / 2, [abs(first_weight_grad), abs(first_bias_grad)], width=width, label="1次 backward")  # 第一次梯度绝对值
plt.bar(positions + width / 2, [abs(second_weight_grad), abs(second_bias_grad)], width=width, label="未清零后第2次")  # 累加后
plt.xticks(positions, ["weight.grad", "bias.grad"], fontproperties=CJK_FONT)  # 横轴标签
plt.ylabel("Gradient 绝对值", fontproperties=CJK_FONT)  # 纵轴
plt.title("不执行 zero_grad：梯度会在 .grad 中累加", fontproperties=CJK_FONT)  # 标题
plt.legend(prop=CJK_FONT)  # 图例
plt.tight_layout()  # 自动排版
plt.savefig(FIG_DIR / "lesson31_gradient_accumulation.png", dpi=180)  # 保存图片
plt.close()  # 关闭画布


# -----------------------------------------------------------------------------
# 实验4：故意省略 backward —— optimizer.step() 没有新梯度可用
# -----------------------------------------------------------------------------
title("实验4：故意省略 backward —— step 能不能凭空知道更新方向？")

no_backward_model = make_model(seed=31)  # 相同初始化
no_backward_optimizer = torch.optim.SGD(no_backward_model.parameters(), lr=0.05)  # SGD
no_backward_optimizer.zero_grad(set_to_none=True)  # 明确让 Gradient 为 None
weight_nb_before = no_backward_model.weight.detach().clone()  # step 前 Weight
bias_nb_before = no_backward_model.bias.detach().clone()  # step 前 Bias
no_backward_optimizer.step()  # 没有 backward，Parameter.grad 仍为 None
weight_nb_after = no_backward_model.weight.detach().clone()  # step 后 Weight
bias_nb_after = no_backward_model.bias.detach().clone()  # step 后 Bias
print("weight.grad =", no_backward_model.weight.grad)  # None
print("bias.grad   =", no_backward_model.bias.grad)  # None
print("未 backward 时 step 修改 weight:", not torch.equal(weight_nb_before, weight_nb_after))  # False
print("未 backward 时 step 修改 bias  :", not torch.equal(bias_nb_before, bias_nb_after))  # False


# -----------------------------------------------------------------------------
# 实验5：故意省略 step —— 梯度算出来了，但参数不更新
# -----------------------------------------------------------------------------
title("实验5：故意省略 step —— loss.backward() 本身不会训练模型")

no_step_model = make_model(seed=31)  # 相同初始化
no_step_criterion = nn.MSELoss()  # MSE
no_step_optimizer = torch.optim.SGD(no_step_model.parameters(), lr=0.05)  # 仅用于 zero_grad

losses_without_step = []  # 保存每轮 Loss
weight_ns_before = no_step_model.weight.detach().clone()  # 初始 Weight
bias_ns_before = no_step_model.bias.detach().clone()  # 初始 Bias

for _ in range(5):  # 重复 5 次
    no_step_optimizer.zero_grad()  # 清梯度
    no_step_pred = no_step_model(X)  # Forward
    no_step_loss = no_step_criterion(no_step_pred, Y)  # Loss
    no_step_loss.backward()  # Gradient 确实算出来了
    losses_without_step.append(no_step_loss.item())  # 记录 Loss
    # 故意不调用 no_step_optimizer.step()

weight_ns_after = no_step_model.weight.detach().clone()  # 结束后的 Weight
bias_ns_after = no_step_model.bias.detach().clone()  # 结束后的 Bias
print("5次 backward 但无 step 的 Loss =", [round(v, 8) for v in losses_without_step])  # 应保持不变
print("weight 是否变化:", not torch.equal(weight_ns_before, weight_ns_after))  # False
print("bias 是否变化  :", not torch.equal(bias_ns_before, bias_ns_after))  # False


# -----------------------------------------------------------------------------
# 实验6：评估阶段 —— torch.no_grad() 为什么常见？
# -----------------------------------------------------------------------------
title("实验6：Evaluation —— torch.no_grad() 不改变 Forward 数值，但不构建反向所需图")

sample_x = torch.tensor([[0.5]], dtype=torch.float32)  # 一个测试输入
train_style_output = train_model(sample_x)  # 默认会被 Autograd 追踪，因为模型参数 requires_grad=True
with torch.no_grad():  # 关闭梯度记录
    eval_output = train_model(sample_x)  # 数值计算仍然正常

print("普通 forward 输出 =", train_style_output.item())  # 普通 Forward 数值
print("no_grad 输出      =", eval_output.item())  # no_grad 数值
print("两者数值差        =", abs(train_style_output.item() - eval_output.item()))  # 应为0
print("普通 forward requires_grad =", train_style_output.requires_grad)  # True
print("no_grad output requires_grad =", eval_output.requires_grad)  # False


# -----------------------------------------------------------------------------
# 实验7：把整个 Training Step 画成生命周期图
# -----------------------------------------------------------------------------
title("实验7：训练 Step 的五阶段职责总结")

stages = ["zero_grad", "forward", "loss", "backward", "step"]  # 五个阶段
notes = [
    "清旧梯度",
    "算预测",
    "量化错误",
    "算Gradient",
    "改Parameter",
]  # 每个阶段的职责

plt.figure(figsize=(9.0, 3.8))  # 创建画布
x_pos = np.arange(len(stages))  # 节点横坐标
plt.scatter(x_pos, np.zeros_like(x_pos), s=900)  # 五个节点
for i, (stage, note) in enumerate(zip(stages, notes)):  # 逐个添加文本
    plt.text(i, 0.0, stage, ha="center", va="center", fontsize=10)  # 节点名称
    plt.text(i, -0.28, note, ha="center", va="top", fontproperties=CJK_FONT, fontsize=10)  # 职责说明
    if i < len(stages) - 1:  # 不是最后一个节点时画箭头
        plt.annotate("", xy=(i + 0.78, 0), xytext=(i + 0.22, 0), arrowprops={"arrowstyle": "->"})  # 顺序箭头
plt.xlim(-0.6, len(stages) - 0.4)  # 横轴范围
plt.ylim(-0.6, 0.45)  # 纵轴范围
plt.axis("off")  # 隐藏坐标轴
plt.title("一个标准 PyTorch Training Step 的职责链", fontproperties=CJK_FONT)  # 标题
plt.tight_layout()  # 自动排版
plt.savefig(FIG_DIR / "lesson31_training_step_flow.png", dpi=180)  # 保存图
plt.close()  # 关闭画布

print("五步顺序:", " -> ".join(stages))  # 打印训练顺序
print("实验完成。图像已保存到:", FIG_DIR)  # 输出图像目录
