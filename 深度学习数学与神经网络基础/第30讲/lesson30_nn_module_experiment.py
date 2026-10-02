# 第30讲实验：nn.Module、Parameter、state_dict 与模型组织
# 目标：不用把 nn.Module 当黑盒，观察一个 PyTorch 模型究竟注册了什么、保存了什么、forward 做了什么。

from pathlib import Path  # 处理输出路径
import sys  # 读取 Python 版本
import numpy as np  # 用于绘图时组织数值
import torch  # PyTorch 主库
import torch.nn as nn  # 神经网络模块
from matplotlib import pyplot as plt  # 绘图
from matplotlib import font_manager  # 中文字体支持

# -----------------------------------------------------------------------------
# 中文绘图字体
# -----------------------------------------------------------------------------
CJK_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"  # 容器中的中文字体
font_manager.fontManager.addfont(CJK_FONT_PATH)  # 注册字体
CJK_FONT = font_manager.FontProperties(fname=CJK_FONT_PATH)  # 创建中文字体对象
plt.rcParams["axes.unicode_minus"] = False  # 避免负号乱码

BASE_DIR = Path(__file__).resolve().parent  # 当前脚本所在目录
FIG_DIR = BASE_DIR / "figures"  # 图像目录
FIG_DIR.mkdir(exist_ok=True)  # 确保目录存在


def title(text):
    """打印清晰的实验分隔标题。"""
    print("\n" + "=" * 86)  # 分隔线
    print(text)  # 标题正文
    print("=" * 86)  # 分隔线


def count_parameters(model):
    """统计模型所有 Parameter 的元素总数。"""
    return sum(p.numel() for p in model.parameters())  # 对每个参数张量的元素数量求和


# -----------------------------------------------------------------------------
# 实验0：环境信息
# -----------------------------------------------------------------------------
title("实验0：环境信息")
print("Python:", sys.version.split()[0])  # Python 版本
print("PyTorch:", torch.__version__)  # PyTorch 版本
print("CUDA available:", torch.cuda.is_available())  # 当前环境是否可以使用 CUDA

torch.manual_seed(30)  # 固定随机种子，保证实验可复现


# -----------------------------------------------------------------------------
# 实验1：定义一个真正的 nn.Module
# -----------------------------------------------------------------------------
title("实验1：定义 TinyMLP —— __init__ 负责搭零件，forward 负责定义数据流")


class TinyMLP(nn.Module):
    """一个 3 -> 4 -> 2 的最小 MLP。"""

    def __init__(self):
        super().__init__()  # 初始化父类 nn.Module，让 PyTorch 能正确注册子模块与参数
        self.fc1 = nn.Linear(3, 4)  # 第一层：3维输入 -> 4维隐藏表示
        self.act = nn.ReLU()  # 激活函数：逐元素 ReLU
        self.fc2 = nn.Linear(4, 2)  # 第二层：4维隐藏表示 -> 2维输出

    def forward(self, x):
        z1 = self.fc1(x)  # 第一步：第一层 Linear
        a1 = self.act(z1)  # 第二步：ReLU
        y = self.fc2(a1)  # 第三步：第二层 Linear
        return y  # 返回模型输出


model = TinyMLP()  # 创建模型实例
print(model)  # PyTorch 会打印模型的模块树
print("模型类型:", type(model).__name__)  # 打印模型类型
print("总参数量:", count_parameters(model))  # 3*4+4 + 4*2+2 = 26


# -----------------------------------------------------------------------------
# 实验2：固定参数，证明 model(x) 与手工调用各层完全一致
# -----------------------------------------------------------------------------
title("实验2：model(x) 与手工 fc2(ReLU(fc1(x))) 是否完全一致？")

W1 = torch.tensor(
    [[0.5, -1.0, 2.0],
     [1.0,  0.5, -0.5],
     [-1.0, 2.0, 0.5],
     [0.2, -0.3, 1.0]],
    dtype=torch.float32,
)  # fc1 的权重，PyTorch 存成 (out_features, in_features)=(4,3)
b1 = torch.tensor([0.1, -0.2, 0.3, 0.0], dtype=torch.float32)  # fc1 的 Bias
W2 = torch.tensor(
    [[1.0, -0.5, 0.25, 2.0],
     [-1.0, 1.5, 0.5, -0.25]],
    dtype=torch.float32,
)  # fc2 的权重，Shape=(2,4)
b2 = torch.tensor([0.2, -0.1], dtype=torch.float32)  # fc2 的 Bias

with torch.no_grad():  # 固定参数只是准备实验，不需要记录到 Autograd 计算图
    model.fc1.weight.copy_(W1)  # 写入第一层 Weight
    model.fc1.bias.copy_(b1)  # 写入第一层 Bias
    model.fc2.weight.copy_(W2)  # 写入第二层 Weight
    model.fc2.bias.copy_(b2)  # 写入第二层 Bias

x = torch.tensor([[1.0, 2.0, -1.0]], dtype=torch.float32)  # 一个样本，Shape=(1,3)
y_model = model(x)  # 推荐用法：通过 model(x) 调用模型
z1_manual = x @ model.fc1.weight.T + model.fc1.bias  # 手工执行第一层 Linear
a1_manual = torch.relu(z1_manual)  # 手工执行 ReLU
y_manual = a1_manual @ model.fc2.weight.T + model.fc2.bias  # 手工执行第二层 Linear

print("输入 x.shape =", tuple(x.shape))  # 输入 Shape
print("fc1.weight.shape =", tuple(model.fc1.weight.shape))  # 第一层 Weight Shape
print("fc1.bias.shape =", tuple(model.fc1.bias.shape))  # 第一层 Bias Shape
print("fc2.weight.shape =", tuple(model.fc2.weight.shape))  # 第二层 Weight Shape
print("fc2.bias.shape =", tuple(model.fc2.bias.shape))  # 第二层 Bias Shape
print("z1_manual =", z1_manual)  # 第一层输出
print("a1_manual =", a1_manual)  # ReLU 后结果
print("model(x) =", y_model)  # Module 前向结果
print("manual   =", y_manual)  # 手工结果
print("最大绝对误差 =", torch.max(torch.abs(y_model - y_manual)).item())  # 应为0

# 绘图：模型输出与手工输出逐项对比
indices = np.arange(y_model.numel())  # 输出位置
width = 0.36  # 柱宽
plt.figure(figsize=(7.2, 4.6))  # 创建画布
plt.bar(indices - width / 2, y_model.detach().numpy().ravel(), width=width, label="model(x)")  # Module 输出
plt.bar(indices + width / 2, y_manual.detach().numpy().ravel(), width=width, label="手工 Forward")  # 手工输出
plt.xticks(indices, ["输出0", "输出1"], fontproperties=CJK_FONT)  # 横轴标签
plt.ylabel("数值", fontproperties=CJK_FONT)  # 纵轴
plt.title("nn.Module 前向结果与手工计算完全一致", fontproperties=CJK_FONT)  # 标题
plt.legend(prop=CJK_FONT)  # 图例
plt.tight_layout()  # 调整布局
plt.savefig(FIG_DIR / "lesson30_module_vs_manual.png", dpi=180)  # 保存图片
plt.close()  # 关闭画布


# -----------------------------------------------------------------------------
# 实验3：named_parameters() —— 模型到底注册了哪些可训练参数？
# -----------------------------------------------------------------------------
title("实验3：named_parameters() —— PyTorch 到底认哪些东西是可训练 Parameter？")

param_names = []  # 保存参数名称，后面用于绘图
param_counts = []  # 保存每个参数的元素数量
for name, parameter in model.named_parameters():  # 遍历已注册 Parameter
    print(
        f"{name:12s} shape={tuple(parameter.shape)!s:10s} "
        f"numel={parameter.numel():2d} requires_grad={parameter.requires_grad}"
    )  # 打印参数的名称、Shape、元素数量与梯度开关
    param_names.append(name)  # 记录名称
    param_counts.append(parameter.numel())  # 记录元素数量

print("model.parameters() 总元素数 =", count_parameters(model))  # 再次统计总参数量
print("按公式计算 = (3*4+4) + (4*2+2) =", (3 * 4 + 4) + (4 * 2 + 2))  # 手工验证

# 绘图：参数量按张量分布
plt.figure(figsize=(8.0, 4.8))  # 创建画布
plt.bar(np.arange(len(param_names)), param_counts)  # 绘制每个参数张量的元素数量
plt.xticks(np.arange(len(param_names)), param_names, rotation=18)  # 参数名称
plt.ylabel("参数元素数量", fontproperties=CJK_FONT)  # 纵轴
plt.title("TinyMLP：Parameter 分布，总计 26 个标量", fontproperties=CJK_FONT)  # 标题
plt.tight_layout()  # 调整布局
plt.savefig(FIG_DIR / "lesson30_parameter_counts.png", dpi=180)  # 保存图片
plt.close()  # 关闭画布


# -----------------------------------------------------------------------------
# 实验4：Parameter、普通 Tensor、Buffer 的区别
# -----------------------------------------------------------------------------
title("实验4：Parameter、普通 Tensor、Buffer —— 三种模型状态有什么不同？")


class StateDemo(nn.Module):
    """演示模型中的三类状态。"""

    def __init__(self):
        super().__init__()  # 初始化 nn.Module
        self.scale = nn.Parameter(torch.tensor(1.5))  # Parameter：可训练、自动注册
        self.plain_tensor = torch.tensor(2.0)  # 普通 Tensor：只是普通 Python 属性
        self.register_buffer("fixed_offset", torch.tensor(3.0))  # Buffer：保存/迁移，但不优化

    def forward(self, x):
        return x * self.scale + self.fixed_offset + self.plain_tensor  # 使用三种状态参与计算


state_demo = StateDemo()  # 创建演示模型
print("named_parameters:", [name for name, _ in state_demo.named_parameters()])  # 只有 scale
print("named_buffers:", [name for name, _ in state_demo.named_buffers()])  # 只有 fixed_offset
print("state_dict keys:", list(state_demo.state_dict().keys()))  # Parameter + Buffer
print("plain_tensor 是否在 state_dict 中:", "plain_tensor" in state_demo.state_dict())  # False
print("scale.requires_grad =", state_demo.scale.requires_grad)  # True
print("fixed_offset.requires_grad =", state_demo.fixed_offset.requires_grad)  # False
print("plain_tensor.requires_grad =", state_demo.plain_tensor.requires_grad)  # False


# -----------------------------------------------------------------------------
# 实验5：state_dict —— 保存的是“模型状态”，不是 forward 代码本身
# -----------------------------------------------------------------------------
title("实验5：state_dict() —— 保存、恢复模型参数后，输出是否完全一致？")

state = model.state_dict()  # 获取模型状态字典
print("TinyMLP state_dict keys:")  # 标题
for key, value in state.items():  # 遍历状态字典
    print(f"  {key:10s} shape={tuple(value.shape)}")  # 打印每个键与 Shape

save_path = BASE_DIR / "lesson30_tiny_mlp_state.pt"  # 状态文件路径
torch.save(model.state_dict(), save_path)  # 只保存 state_dict

loaded_model = TinyMLP()  # 创建结构相同但参数随机的新模型
output_before_load = loaded_model(x).detach().clone()  # 记录加载前输出
loaded_model.load_state_dict(torch.load(save_path, map_location="cpu", weights_only=True))  # 加载状态
output_after_load = loaded_model(x).detach().clone()  # 记录加载后输出

print("新模型加载前输出 =", output_before_load)  # 通常不同
print("加载 state_dict 后输出 =", output_after_load)  # 应与原模型一致
print("原模型输出 =", y_model.detach())  # 原模型结果
print("加载后与原模型最大绝对误差 =", torch.max(torch.abs(output_after_load - y_model.detach())).item())  # 应为0


# -----------------------------------------------------------------------------
# 实验6：model.to(device) —— Module 会统一迁移已注册 Parameter 与 Buffer
# -----------------------------------------------------------------------------
title("实验6：model.to(device) —— 为什么 Module 能统一移动参数和 Buffer？")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择设备
state_demo = state_demo.to(device)  # 一次移动整个 Module
print("选中的 device =", device)  # 打印设备
print("Parameter scale.device =", state_demo.scale.device)  # Parameter 会随模型移动
print("Buffer fixed_offset.device =", state_demo.fixed_offset.device)  # Buffer 也会随模型移动
print("普通 plain_tensor.device =", state_demo.plain_tensor.device)  # 普通属性不会被 Module 机制主动管理

# 为了保证 CPU-only 和 GPU 环境都能执行 forward，手工把普通 Tensor 对齐到 device。
state_demo.plain_tensor = state_demo.plain_tensor.to(device)  # 普通 Tensor 需要我们自己处理
input_device = torch.tensor(4.0, device=device)  # 构造同设备输入
print("StateDemo forward 输出 =", state_demo(input_device).item())  # 4*1.5+3+2=11


# -----------------------------------------------------------------------------
# 实验7：model(x) 与 model.forward(x) —— 为什么习惯上应该调用 model(x)？
# -----------------------------------------------------------------------------
title("实验7：model(x) 与 model.forward(x) 的数值结果一致，但推荐使用 model(x)")

out_call = model(x)  # 通过 Module.__call__ 入口执行
out_direct = model.forward(x)  # 直接调用 forward，仅用于演示
print("model(x) =", out_call)  # 数值输出
print("model.forward(x) =", out_direct)  # 数值输出
print("当前最小例子最大绝对误差 =", torch.max(torch.abs(out_call - out_direct)).item())  # 应为0
print("说明：推荐 model(x)，因为 __call__ 还负责 hooks 等 Module 机制，再由它调用 forward。")  # 概念说明


# -----------------------------------------------------------------------------
# 实验8：一个 Parameter 做 backward —— 梯度会自动落在注册参数的 .grad 中
# -----------------------------------------------------------------------------
title("实验8：注册 Parameter 怎样接收梯度？")

model.zero_grad()  # 先把模型所有参数梯度清零
pred8 = model(x)  # Forward
loss8 = pred8.sum()  # 构造一个简单标量 Loss
loss8.backward()  # Autograd 反向传播

for name, parameter in model.named_parameters():  # 查看每个 Parameter 的 .grad
    print(f"{name:12s} grad.shape={tuple(parameter.grad.shape)} grad_norm={parameter.grad.norm().item():.6f}")  # 打印梯度 Shape 与范数

# -----------------------------------------------------------------------------
# 实验总结
# -----------------------------------------------------------------------------
title("实验总结")
print("1) __init__ 注册 Layer/Parameter；forward 定义数据怎样流过这些组件。")  # 总结1
print("2) model(x) 数值上等价于本例手工 Forward，但它经过 Module.__call__ 机制。")  # 总结2
print("3) named_parameters() 只列出已注册 Parameter；TinyMLP 总参数量为26。")  # 总结3
print("4) state_dict 包含 Parameter 和持久 Buffer，但不包含普通 Tensor 属性。")  # 总结4
print("5) model.to(device) 能统一迁移已注册 Parameter/Buffer；普通 Tensor 属性需自行管理。")  # 总结5
print("6) backward 后，注册 Parameter 的梯度自动写入 parameter.grad。")  # 总结6
