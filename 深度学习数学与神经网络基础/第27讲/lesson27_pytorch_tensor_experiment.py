# 第27讲实验：PyTorch Tensor —— 从 NumPy 数组进入自动微分框架
# 运行方式：python lesson27_pytorch_tensor_experiment.py

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 显式加载系统中的简体中文字体，避免图中文字变成方块。
CJK_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(CJK_FONT_PATH)
CJK_FONT = font_manager.FontProperties(fname=CJK_FONT_PATH)
plt.rcParams["axes.unicode_minus"] = False

# 取得当前脚本所在目录，后续把图片固定保存到 figures 子目录。
BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)


def title(text):
    """打印实验小节标题。"""
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


def show_meta(name, obj):
    """统一打印 NumPy / PyTorch 对象的关键元数据。"""
    if isinstance(obj, np.ndarray):
        print(f"{name}: type={type(obj).__name__}, shape={obj.shape}, ndim={obj.ndim}, dtype={obj.dtype}")
    elif isinstance(obj, torch.Tensor):
        print(
            f"{name}: type={type(obj).__name__}, shape={tuple(obj.shape)}, "
            f"ndim={obj.ndim}, dtype={obj.dtype}, device={obj.device}, "
            f"requires_grad={obj.requires_grad}"
        )
    else:
        print(f"{name}: type={type(obj).__name__}")


# ---------------------------------------------------------------------------
# 实验0：环境信息
# ---------------------------------------------------------------------------
title("实验0：环境与版本")
print("Python:", sys.version.split()[0])
print("NumPy:", np.__version__)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# ---------------------------------------------------------------------------
# 实验1：同一组数字，用 ndarray 和 Tensor 分别表示
# ---------------------------------------------------------------------------
title("实验1：NumPy ndarray 与 torch.Tensor 的基本对应")

# 用 NumPy 创建一个 2×3 的浮点数组。
np_x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)

# 用 PyTorch 创建完全相同数值和 dtype 的 Tensor。
torch_x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)

# 打印二者的元数据。
show_meta("np_x", np_x)
show_meta("torch_x", torch_x)

# 比较二者数值是否完全一致。
print("values equal:", np.allclose(np_x, torch_x.numpy()))

# ---------------------------------------------------------------------------
# 实验2：索引、切片与 Shape —— 规则基本延续 NumPy
# ---------------------------------------------------------------------------
title("实验2：索引、切片与 Shape")

# 取第0行：会移除第0维，因此 Shape 由 (2,3) 变为 (3,)。
row0 = torch_x[0]

# 用切片取第0行：保留第0维，因此 Shape 是 (1,3)。
row0_keepdim = torch_x[0:1]

# 取第1列：Shape 为 (2,)。
col1 = torch_x[:, 1]

# 取第1列但保留列维：Shape 为 (2,1)。
col1_keepdim = torch_x[:, 1:2]

show_meta("torch_x[0]", row0)
show_meta("torch_x[0:1]", row0_keepdim)
show_meta("torch_x[:,1]", col1)
show_meta("torch_x[:,1:2]", col1_keepdim)

# ---------------------------------------------------------------------------
# 实验3：逐元素乘法与矩阵乘法是两件不同的事
# ---------------------------------------------------------------------------
title("实验3：elementwise '*' 与 matrix multiplication '@'")

# 创建一个和 torch_x Shape 相同的 Tensor。
B = torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])

# '*' 是对应位置逐元素相乘，Shape 保持 (2,3)。
elementwise = torch_x * B

# 为矩阵乘法创建一个 (3,2) 权重矩阵。
W = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]])

# '@' 做矩阵乘法：(2,3)@(3,2) -> (2,2)。
matmul = torch_x @ W

print("elementwise =\n", elementwise)
show_meta("elementwise", elementwise)
print("matmul =\n", matmul)
show_meta("matmul", matmul)

# ---------------------------------------------------------------------------
# 实验4：dtype —— NumPy 与 PyTorch 的默认浮点类型可能不同
# ---------------------------------------------------------------------------
title("实验4：dtype 与显式类型控制")

# Python 浮点数创建 NumPy 数组时，NumPy 通常默认 float64。
np_default = np.array([1.0, 2.0, 3.0])

# Python 浮点数创建 PyTorch Tensor 时，PyTorch 通常默认 float32。
torch_default = torch.tensor([1.0, 2.0, 3.0])

show_meta("np_default", np_default)
show_meta("torch_default", torch_default)

# 显式转换成 float64。
torch_float64 = torch_default.to(torch.float64)
show_meta("torch_float64", torch_float64)

# ---------------------------------------------------------------------------
# 实验5：torch.from_numpy 与 torch.tensor 的内存行为
# ---------------------------------------------------------------------------
title("实验5：from_numpy 共享 CPU 内存；torch.tensor 通常复制数据")

# 创建一个 float32 NumPy 数组。
a = np.array([1.0, 2.0, 3.0], dtype=np.float32)

# from_numpy 在 CPU 上通常与原数组共享底层内存。
t_shared = torch.from_numpy(a)

# torch.tensor(a) 通常会复制出一个新的 Tensor。
t_copy = torch.tensor(a)

print("初始:")
print("numpy    =", a)
print("shared   =", t_shared)
print("copy     =", t_copy)

# 修改 NumPy 数组中的第0个元素。
a[0] = 99.0
print("\n修改 a[0] = 99 后:")
print("numpy    =", a)
print("shared   =", t_shared)
print("copy     =", t_copy)

# 再修改共享 Tensor 的第1个元素。
t_shared[1] = 77.0
print("\n修改 t_shared[1] = 77 后:")
print("numpy    =", a)
print("shared   =", t_shared)
print("copy     =", t_copy)

# 保存一张图，直观看内存共享与复制的差异。
stages = ["初始", "NumPy修改后", "shared修改后"]
np_first = [1.0, 99.0, 99.0]
shared_first = [1.0, 99.0, 99.0]
copy_first = [1.0, 1.0, 1.0]

plt.figure(figsize=(8, 4.8))
plt.plot(stages, np_first, marker="o", label="NumPy a[0]")
plt.plot(stages, shared_first, marker="s", label="from_numpy Tensor[0]")
plt.plot(stages, copy_first, marker="^", label="torch.tensor copy[0]")
plt.ylabel("第0个元素的值", fontproperties=CJK_FONT)
plt.title("from_numpy 共享内存 vs torch.tensor 复制", fontproperties=CJK_FONT)
plt.xticks(fontproperties=CJK_FONT)
plt.legend(prop=CJK_FONT)
plt.tight_layout()
plt.savefig(FIG_DIR / "lesson27_memory_sharing.png", dpi=180)
plt.close()

# ---------------------------------------------------------------------------
# 实验6：device —— Tensor 在哪里计算
# ---------------------------------------------------------------------------
title("实验6：device —— CPU / GPU 是 Tensor 的一部分元数据")

# 如果当前机器有 CUDA，就使用 GPU；否则使用 CPU。
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 把 Tensor 移动到选择的 device。
x_device = torch_x.to(device)

print("selected device:", device)
show_meta("x_device", x_device)

# ---------------------------------------------------------------------------
# 实验7：requires_grad —— 什么时候 Tensor 开始进入自动微分计算图
# ---------------------------------------------------------------------------
title("实验7：requires_grad 与计算图预览（本讲不正式调用 backward）")

# 这是一个需要求梯度的叶子 Tensor。
x_grad = torch.tensor([2.0, -1.0], dtype=torch.float32, requires_grad=True)

# 只要用它做普通 PyTorch 运算，结果就会记录梯度依赖关系。
y_grad = x_grad * 3.0 + 1.0
z_grad = y_grad ** 2

show_meta("x_grad", x_grad)
show_meta("y_grad", y_grad)
show_meta("z_grad", z_grad)
print("x_grad.is_leaf:", x_grad.is_leaf)
print("y_grad.is_leaf:", y_grad.is_leaf)
print("z_grad.grad_fn:", type(z_grad.grad_fn).__name__)
print("注意：此时还没有调用 backward()，所以 x_grad.grad 仍然是:", x_grad.grad)

# no_grad 会暂时关闭梯度跟踪，常用于推理阶段。
with torch.no_grad():
    y_no_grad = x_grad * 3.0 + 1.0

show_meta("y_no_grad", y_no_grad)

# ---------------------------------------------------------------------------
# 实验8：LLM 风格 (B,T,D) Tensor 继续使用同一套矩阵规则
# ---------------------------------------------------------------------------
title("实验8：LLM 风格 (B,T,D) Tensor 与 Linear-like 矩阵乘法")

# 构造 B=2, T=3, D=4 的三维 Tensor。
X = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)

# 构造一个从 D=4 映射到 O=5 的权重矩阵。
W_llm = torch.arange(4 * 5, dtype=torch.float32).reshape(4, 5) / 10.0

# PyTorch 的 matmul 会把最后两维按矩阵规则计算，前面的 B、T 维保留。
Y = X @ W_llm

show_meta("X", X)
show_meta("W_llm", W_llm)
show_meta("Y = X @ W_llm", Y)
print("Shape rule: (B,T,D)@(D,O) -> (B,T,O)")

# ---------------------------------------------------------------------------
# 实验9：NumPy <-> Tensor 转换与 detach 的最小预览
# ---------------------------------------------------------------------------
title("实验9：Tensor 转 NumPy 与 detach 的最小预览")

# 普通 CPU Tensor 可以直接转成 NumPy。
normal_tensor = torch.tensor([1.0, 2.0, 3.0])
normal_numpy = normal_tensor.numpy()
print("normal_tensor.numpy() =", normal_numpy)

# 参与梯度跟踪的 Tensor 应先 detach，再移到 CPU，再转 NumPy。
tracked = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
tracked_numpy = tracked.detach().cpu().numpy()
print("tracked.detach().cpu().numpy() =", tracked_numpy)

# ---------------------------------------------------------------------------
# 结束总结
# ---------------------------------------------------------------------------
title("实验总结")
print("1) torch.Tensor 仍然是多维数字容器，但多了 dtype/device/autograd 等能力。")
print("2) NumPy 与 PyTorch 的 Shape、索引、矩阵乘法直觉高度一致。")
print("3) dtype 和 device 是深度学习实验中必须检查的元数据。")
print("4) from_numpy 可能共享 CPU 内存；torch.tensor 通常复制。")
print("5) requires_grad=True 会让 Tensor 开始参与自动微分计算图。")
print("6) 本讲只预览 autograd；正式 backward() 会在第29讲系统学习。")
