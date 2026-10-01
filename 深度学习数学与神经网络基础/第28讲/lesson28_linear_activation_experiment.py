# 第28讲实验：PyTorch 中的 Linear 与 Activation
# 运行方式：python lesson28_linear_activation_experiment.py

from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ------------------------------
# 中文字体与输出目录
# ------------------------------
CJK_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(CJK_FONT_PATH)
CJK_FONT = font_manager.FontProperties(fname=CJK_FONT_PATH)
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)


def title(text):
    """打印清晰的实验分隔标题。"""
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


def count_parameters(module):
    """统计一个 PyTorch 模块中的全部可训练参数数量。"""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# -----------------------------------------------------------------------------
# 实验0：环境信息与可复现设置
# -----------------------------------------------------------------------------
title("实验0：环境信息与随机种子")
print("Python:", sys.version.split()[0])
print("NumPy:", np.__version__)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# 固定随机种子，保证默认初始化相关结果可以复现。
torch.manual_seed(28)
np.random.seed(28)

# -----------------------------------------------------------------------------
# 实验1：nn.Linear 到底保存了什么？
# -----------------------------------------------------------------------------
title("实验1：检查 nn.Linear(3, 2) 的 Weight、Bias、Shape 与参数量")

# 创建一个输入维度为3、输出维度为2的线性层。
linear = nn.Linear(in_features=3, out_features=2, bias=True)

print(linear)
print("weight.shape =", tuple(linear.weight.shape))
print("bias.shape   =", tuple(linear.bias.shape))
print("weight.requires_grad =", linear.weight.requires_grad)
print("bias.requires_grad   =", linear.bias.requires_grad)
print("trainable parameter count =", count_parameters(linear))
print("理论参数量 = 2*3 + 2 =", 2 * 3 + 2)

# 打印参数名和Shape，建立“模块里包含参数”的直觉。
for name, parameter in linear.named_parameters():
    print(f"named_parameter: {name:6s} shape={tuple(parameter.shape)} numel={parameter.numel()}")

# -----------------------------------------------------------------------------
# 实验2：把 nn.Linear 与手写 X @ W.T + b 完全对齐
# -----------------------------------------------------------------------------
title("实验2：手写 X @ W.T + b 与 nn.Linear 数值完全对齐")

# 为了便于手算，定义确定的 Weight 和 Bias。
# 注意：PyTorch 保存的 weight Shape 是 (out_features, in_features) = (2, 3)。
known_weight = torch.tensor(
    [[0.5, -1.0, 2.0],
     [1.0,  0.5, -0.5]],
    dtype=torch.float32,
)
known_bias = torch.tensor([0.2, -0.3], dtype=torch.float32)

# 参数默认 requires_grad=True，直接原地 copy_ 会被 Autograd 阻止；
# 所以只在“给参数赋固定测试值”时临时关闭梯度记录。
with torch.no_grad():
    linear.weight.copy_(known_weight)
    linear.bias.copy_(known_bias)

# 单样本使用一维 Tensor，Shape 为 (3,)。
x = torch.tensor([1.0, 2.0, -1.0], dtype=torch.float32)

# 手工按照 PyTorch 存储约定计算：x @ weight.T + bias。
y_manual = x @ linear.weight.T + linear.bias

# 直接调用模块。
y_module = linear(x)

print("x =", x)
print("weight =\n", linear.weight)
print("bias =", linear.bias)
print("manual y =", y_manual)
print("module y =", y_module)
print("max abs diff =", torch.max(torch.abs(y_manual - y_module)).item())
print("手算期望结果约为 [-3.3, 2.2]")

# -----------------------------------------------------------------------------
# 实验3：Batch 输入 —— nn.Linear 作用在最后一个维度
# -----------------------------------------------------------------------------
title("实验3：二维 Batch 输入 (B, Din) -> (B, Dout)")

# 三个样本，每个样本有3个特征。
X_batch = torch.tensor(
    [[1.0, 2.0, -1.0],
     [0.0, 1.0,  2.0],
     [2.0, 0.0,  1.0]],
    dtype=torch.float32,
)

# 模块一次处理整个Batch。
Y_batch_module = linear(X_batch)

# 手工使用矩阵乘法对照。
Y_batch_manual = X_batch @ linear.weight.T + linear.bias

print("X_batch.shape =", tuple(X_batch.shape))
print("Y_batch.shape =", tuple(Y_batch_module.shape))
print("Y_batch =\n", Y_batch_module)
print("batch max abs diff =", torch.max(torch.abs(Y_batch_manual - Y_batch_module)).item())

# -----------------------------------------------------------------------------
# 实验4：LLM 风格三维输入 —— 前导维保留，最后一维被映射
# -----------------------------------------------------------------------------
title("实验4：三维 Tensor (B, T, Din) -> (B, T, Dout)")

# B=2条序列，T=4个Token，每个Token有3维特征。
X_llm = torch.tensor(
    [
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
        [[2.0, 0.0, 1.0], [1.0, 2.0, 0.0], [-1.0, 1.0, 1.0], [0.5, 0.5, 0.5]],
    ],
    dtype=torch.float32,
)

# nn.Linear 会自动把最后一维3映射到2，并保留 B 与 T。
Y_llm = linear(X_llm)

# 手工公式仍然一样：最后一维与 weight.T 做矩阵乘法。
Y_llm_manual = X_llm @ linear.weight.T + linear.bias

print("X_llm.shape =", tuple(X_llm.shape))
print("Y_llm.shape =", tuple(Y_llm.shape))
print("3D max abs diff =", torch.max(torch.abs(Y_llm_manual - Y_llm)).item())
print("第0条序列第0个Token输出 =", Y_llm[0, 0])
print("Shape rule: (B,T,Din) -> Linear(Din,Dout) -> (B,T,Dout)")

# -----------------------------------------------------------------------------
# 实验5：Activation 改变数值，但逐元素激活通常保持 Shape
# -----------------------------------------------------------------------------
title("实验5：ReLU / GELU / SiLU 的输入输出与 Shape")

z = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=torch.float32)

relu = nn.ReLU()
gelu = nn.GELU()
silu = nn.SiLU()

relu_out = relu(z)
gelu_out = gelu(z)
silu_out = silu(z)

print("z        =", z)
print("ReLU(z)  =", relu_out)
print("GELU(z)  =", gelu_out)
print("SiLU(z)  =", silu_out)
print("input shape =", tuple(z.shape))
print("ReLU shape =", tuple(relu_out.shape))
print("GELU shape =", tuple(gelu_out.shape))
print("SiLU shape =", tuple(silu_out.shape))

# 验证 module 写法和 functional 写法的 ReLU 结果一致。
relu_functional = F.relu(z)
print("nn.ReLU vs F.relu max diff =", torch.max(torch.abs(relu_out - relu_functional)).item())

# 绘制三种激活函数曲线，帮助建立数值直觉。
x_curve = torch.linspace(-5.0, 5.0, 501)
with torch.no_grad():
    y_relu = relu(x_curve).numpy()
    y_gelu = gelu(x_curve).numpy()
    y_silu = silu(x_curve).numpy()

plt.figure(figsize=(8.4, 5.2))
plt.plot(x_curve.numpy(), y_relu, label="ReLU")
plt.plot(x_curve.numpy(), y_gelu, label="GELU")
plt.plot(x_curve.numpy(), y_silu, label="SiLU")
plt.axhline(0.0, linewidth=0.8)
plt.axvline(0.0, linewidth=0.8)
plt.xlabel("输入 z", fontproperties=CJK_FONT)
plt.ylabel("激活后输出", fontproperties=CJK_FONT)
plt.title("ReLU / GELU / SiLU：相同 Shape，不同数值映射", fontproperties=CJK_FONT)
plt.legend(prop=CJK_FONT)
plt.tight_layout()
plt.savefig(FIG_DIR / "lesson28_activation_curves.png", dpi=180)
plt.close()

# -----------------------------------------------------------------------------
# 实验6：Linear + Activation —— PyTorch 中的最小非线性块
# -----------------------------------------------------------------------------
title("实验6：Linear + ReLU 的最小非线性块")

# 建立一个从3维输入映射到2维，再经过ReLU的最小模块。
block = nn.Sequential(
    nn.Linear(3, 2),
    nn.ReLU(),
)

# 把第一层参数设置为和实验2相同，以便结果可解释。
with torch.no_grad():
    block[0].weight.copy_(known_weight)
    block[0].bias.copy_(known_bias)

z_block = block[0](x)
a_block = block(x)

print(block)
print("Linear output z =", z_block)
print("After ReLU a =", a_block)
print("block trainable parameter count =", count_parameters(block))
print("注意：ReLU 本身没有可训练 Weight/Bias，因此参数量仍然是8")

# 保存一个数值对比图：Linear前后与ReLU后。
labels = ["Linear输出1", "Linear输出2"]
linear_values = z_block.detach().numpy()
relu_values = a_block.detach().numpy()
positions = np.arange(len(labels))
width = 0.34

plt.figure(figsize=(7.4, 4.8))
plt.bar(positions - width / 2, linear_values, width=width, label="Linear输出")
plt.bar(positions + width / 2, relu_values, width=width, label="ReLU后")
plt.xticks(positions, labels, fontproperties=CJK_FONT)
plt.ylabel("数值", fontproperties=CJK_FONT)
plt.title("Linear 产生特征，Activation 对特征做非线性映射", fontproperties=CJK_FONT)
plt.legend(prop=CJK_FONT)
plt.tight_layout()
plt.savefig(FIG_DIR / "lesson28_linear_relu_values.png", dpi=180)
plt.close()

# -----------------------------------------------------------------------------
# 实验7：nn.Sequential 与 state_dict —— 模块到底保存了哪些参数？
# -----------------------------------------------------------------------------
title("实验7：nn.Sequential、state_dict 与参数Shape")

# 构造一个最小 PyTorch MLP：3 -> 4 -> 2。
mlp = nn.Sequential(
    nn.Linear(3, 4),
    nn.GELU(),
    nn.Linear(4, 2),
)

print(mlp)
print("trainable parameter count =", count_parameters(mlp))
print("理论参数量 = (3*4+4) + (4*2+2) =", (3 * 4 + 4) + (4 * 2 + 2))

# state_dict 是模型参数与缓冲区的有序映射，是保存/加载模型的重要入口。
print("state_dict entries:")
for name, tensor in mlp.state_dict().items():
    print(f"  {name:10s} shape={tuple(tensor.shape)}")

# 对一个二维Batch进行Forward，确认输出Shape。
X_small = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]], dtype=torch.float32)
Y_small = mlp(X_small)
print("X_small.shape =", tuple(X_small.shape))
print("Y_small.shape =", tuple(Y_small.shape))

# -----------------------------------------------------------------------------
# 实验8：Transformer FFN 风格 Shape —— Linear + GELU + Linear
# -----------------------------------------------------------------------------
title("实验8：Transformer FFN 风格 Shape 追踪")

# 为了资源很小，使用 D=4、hidden=8，而不是实际大模型中的几千维。
ffn = nn.Sequential(
    nn.Linear(4, 8),
    nn.GELU(),
    nn.Linear(8, 4),
)

# 构造 B=2、T=3、D=4 的 hidden states。
hidden = torch.randn(2, 3, 4)

# 分步骤执行，打印每一步Shape。
h1 = ffn[0](hidden)
h2 = ffn[1](h1)
h3 = ffn[2](h2)

print("input hidden shape  =", tuple(hidden.shape))
print("after Linear(4,8)  =", tuple(h1.shape))
print("after GELU         =", tuple(h2.shape))
print("after Linear(8,4)  =", tuple(h3.shape))
print("FFN output shape   =", tuple(ffn(hidden).shape))
print("FFN parameter count =", count_parameters(ffn))

# 画一个简单的Shape流程图，作为实验结果记录。
stages = ["输入", "Linear 4→8", "GELU", "Linear 8→4"]
last_dims = [4, 8, 8, 4]
plt.figure(figsize=(8.2, 4.7))
plt.plot(stages, last_dims, marker="o")
plt.ylabel("最后一个维度长度", fontproperties=CJK_FONT)
plt.title("Transformer FFN 风格：B、T 保留，最后一维先扩张再缩回", fontproperties=CJK_FONT)
plt.xticks(fontproperties=CJK_FONT)
plt.tight_layout()
plt.savefig(FIG_DIR / "lesson28_ffn_shape_flow.png", dpi=180)
plt.close()

# -----------------------------------------------------------------------------
# 最终总结
# -----------------------------------------------------------------------------
title("实验总结")
print("1. nn.Linear(Din, Dout) 内部 weight.shape = (Dout, Din)。")
print("2. 前向计算等价于 X @ weight.T + bias。")
print("3. 对 (B,T,Din) 输入，nn.Linear 只映射最后一维，得到 (B,T,Dout)。")
print("4. ReLU/GELU/SiLU 等逐元素激活通常保持 Shape，但改变数值并引入非线性。")
print("5. nn.Sequential 可以把 Layer 按顺序组合；state_dict 可以查看需要保存的参数。")
print("6. 本讲只做 Forward 与模块检查；正式 Autograd 与 loss.backward() 留到第29讲。")
