# -*- coding: utf-8 -*-
"""
第48讲实验：CNN 基础——Kernel、Convolution、Feature Map 与参数共享
面向 LLM / Agent Systems 科研的深度学习零基础课

实验目标：
1. 用 NumPy 手写二维“卷积”（深度学习框架实际使用 cross-correlation / 互相关）。
2. 与 PyTorch F.conv2d 做数值对照，理解 Kernel 滑动与 Feature Map。
3. 验证 Padding / Stride 对输出 Shape 的影响。
4. 计算 Conv2d 的参数量，理解参数共享为什么极大减少参数。
5. 验证多 Channel 卷积：一个输出 Channel 会汇总所有输入 Channel。
6. 手算并验证 Max Pooling。
7. 观察局部平移后 Feature Map 也相应平移，建立 translation equivariance 直觉。
8. 用一个 TinyCNN 跟踪完整 Shape 与参数量。
"""

from pathlib import Path
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import font_manager

# -----------------------------
# 0. 固定随机种子，保证实验可复现
# -----------------------------
SEED = 48
np.random.seed(SEED)
torch.manual_seed(SEED)

# -----------------------------
# 1. 输出目录
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# 2. Matplotlib 中文字体
# -----------------------------
font_candidates = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
]
for font_path in font_candidates:
    if Path(font_path).exists():
        font_manager.fontManager.addfont(font_path)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False


def print_section(title: str) -> None:
    """打印实验分隔标题。"""
    print("\n" + "=" * 82)
    print(title)
    print("=" * 82)


def xcorr2d_numpy(x: np.ndarray, kernel: np.ndarray, stride: int = 1, padding: int = 0) -> np.ndarray:
    """
    手写二维 cross-correlation（互相关）。

    注意：PyTorch Conv2d 的 Kernel 不翻转，因此数学上严格说更接近 cross-correlation。
    但深度学习社区通常仍称其为 convolution。

    输入：
        x:      (H, W)
        kernel: (K_h, K_w)
    输出：
        out:    (H_out, W_out)
    """
    if padding > 0:
        x_pad = np.pad(x, ((padding, padding), (padding, padding)), mode="constant")
    else:
        x_pad = x.copy()

    h, w = x_pad.shape
    kh, kw = kernel.shape
    h_out = (h - kh) // stride + 1
    w_out = (w - kw) // stride + 1
    out = np.zeros((h_out, w_out), dtype=np.float32)

    for i in range(h_out):
        for j in range(w_out):
            r0 = i * stride
            c0 = j * stride
            patch = x_pad[r0:r0 + kh, c0:c0 + kw]
            out[i, j] = np.sum(patch * kernel)
    return out


def conv2d_multi_channel_numpy(x: np.ndarray, weight: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
    """
    手写多 Channel Conv2d，stride=1, padding=0。

    x.shape      = (C_in, H, W)
    weight.shape = (C_out, C_in, K_h, K_w)
    bias.shape   = (C_out,)
    output.shape = (C_out, H_out, W_out)
    """
    c_in, h, w = x.shape
    c_out, c_in_w, kh, kw = weight.shape
    assert c_in == c_in_w
    h_out = h - kh + 1
    w_out = w - kw + 1
    out = np.zeros((c_out, h_out, w_out), dtype=np.float32)

    for co in range(c_out):
        for i in range(h_out):
            for j in range(w_out):
                total = 0.0
                for ci in range(c_in):
                    patch = x[ci, i:i + kh, j:j + kw]
                    total += float(np.sum(patch * weight[co, ci]))
                if bias is not None:
                    total += float(bias[co])
                out[co, i, j] = total
    return out


# ============================================================================
# 实验1：5x5 输入 + 3x3 Kernel，手工滑动得到 Feature Map
# ============================================================================
print_section("实验1：手写二维卷积（cross-correlation）并与 PyTorch 对照")

# 构造一张非常小的“灰度图”：右半边为 1，形成明显的垂直边缘。
image = np.array([
    [0, 0, 0, 1, 1],
    [0, 0, 0, 1, 1],
    [0, 0, 0, 1, 1],
    [0, 0, 0, 1, 1],
    [0, 0, 0, 1, 1],
], dtype=np.float32)

# 一个简单的垂直边缘检测 Kernel。
kernel = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1],
], dtype=np.float32)

manual_feature = xcorr2d_numpy(image, kernel)

# PyTorch F.conv2d 输入必须有 Batch 和 Channel 维：(N, C, H, W)。
x_t = torch.tensor(image).view(1, 1, 5, 5)
w_t = torch.tensor(kernel).view(1, 1, 3, 3)
torch_feature = F.conv2d(x_t, w_t).squeeze().numpy()

print("输入 image shape =", image.shape)
print(image)
print("Kernel shape =", kernel.shape)
print(kernel)
print("手写 Feature Map shape =", manual_feature.shape)
print(manual_feature)
print("PyTorch Feature Map =")
print(torch_feature)
print("最大绝对误差 =", float(np.max(np.abs(manual_feature - torch_feature))))

# 第一个输出元素的完整数字过程。
first_patch = image[0:3, 0:3]
first_value = float(np.sum(first_patch * kernel))
print("第一个 3x3 patch =")
print(first_patch)
print("第一个输出值 = sum(patch * kernel) =", first_value)

# 图1：输入、Kernel、Feature Map。
fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
for ax, data, title in zip(
    axes,
    [image, kernel, manual_feature],
    ["输入 5×5", "Kernel 3×3", "Feature Map 3×3"],
):
    im = ax.imshow(data, cmap="viridis")
    ax.set_title(title)
    ax.set_xticks(range(data.shape[1]))
    ax.set_yticks(range(data.shape[0]))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i,j]:.0f}", ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.suptitle("Kernel 滑动后得到 Feature Map")
fig.tight_layout()
fig.savefig(FIG_DIR / "lesson48_kernel_feature_map.png", dpi=180)
plt.close(fig)


# ============================================================================
# 实验2：严格数学 convolution 会翻转 Kernel；PyTorch 默认不翻转
# ============================================================================
print_section("实验2：为什么深度学习里的 Conv2d 严格说更接近 cross-correlation")

asym_kernel = np.array([
    [1, 2, 3],
    [0, 1, 0],
    [-1, 0, 2],
], dtype=np.float32)

xcorr_out = xcorr2d_numpy(image, asym_kernel)
flipped_kernel = np.flip(asym_kernel, axis=(0, 1)).copy()
true_conv_out = xcorr2d_numpy(image, flipped_kernel)
pt_out = F.conv2d(x_t, torch.tensor(asym_kernel).view(1, 1, 3, 3)).squeeze().numpy()

print("不翻转 Kernel 的 cross-correlation 与 PyTorch 最大误差 =", float(np.max(np.abs(xcorr_out - pt_out))))
print("严格数学 convolution（翻转 Kernel）与 PyTorch 最大差异 =", float(np.max(np.abs(true_conv_out - pt_out))))
print("结论：PyTorch Conv2d 默认不翻转 Kernel，但深度学习社区仍沿用 convolution 这个名称。")


# ============================================================================
# 实验3：Padding / Stride 与输出 Shape
# ============================================================================
print_section("实验3：Padding / Stride 如何决定输出 Shape")

x_shape = torch.randn(1, 1, 7, 7)
shape_cases = [
    (3, 1, 0),  # kernel, stride, padding
    (3, 1, 1),
    (3, 2, 0),
    (3, 2, 1),
    (5, 1, 0),
]

for k, s, p in shape_cases:
    weight = torch.ones(1, 1, k, k)
    out = F.conv2d(x_shape, weight, stride=s, padding=p)
    h_formula = math.floor((7 + 2 * p - k) / s) + 1
    print(
        f"K={k}, stride={s}, padding={p} -> 公式 H_out={h_formula}, "
        f"PyTorch output shape={tuple(out.shape)}"
    )


# ============================================================================
# 实验4：参数共享为什么能大幅减少参数量
# ============================================================================
print_section("实验4：Parameter Sharing——Conv2d 参数量为什么与 H/W 无关")

conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1, bias=True)
conv_params = sum(p.numel() for p in conv.parameters())
conv_formula = 16 * 3 * 3 * 3 + 16

# 假设用一个全连接层直接把 32x32x3 映射到 32x32x16。
dense_in = 32 * 32 * 3
dense_out = 32 * 32 * 16
dense_params = dense_in * dense_out + dense_out
ratio = dense_params / conv_params

sample_img = torch.randn(1, 3, 32, 32)
conv_out = conv(sample_img)

print("Conv2d weight shape =", tuple(conv.weight.shape))
print("Conv2d bias shape =", tuple(conv.bias.shape))
print("Conv2d 参数量（实际）=", conv_params)
print("Conv2d 参数量（公式）=", conv_formula)
print("输入 shape =", tuple(sample_img.shape))
print("输出 shape =", tuple(conv_out.shape))
print("假设 Dense 直接映射 3072 -> 16384，参数量 =", dense_params)
print(f"Dense / Conv 参数量比例 ≈ {ratio:.1f} 倍")

# 图2：参数量对比（对数轴）。
fig = plt.figure(figsize=(7, 4.5))
plt.bar(["Conv2d\n3→16, 3×3", "Dense\n3072→16384"], [conv_params, dense_params])
plt.yscale("log")
plt.ylabel("参数量（log scale）")
plt.title("参数共享：相同 Kernel 在所有空间位置重复使用")
for i, v in enumerate([conv_params, dense_params]):
    plt.text(i, v * 1.25, f"{v:,}", ha="center")
plt.tight_layout()
fig.savefig(FIG_DIR / "lesson48_parameter_sharing.png", dpi=180)
plt.close(fig)


# ============================================================================
# 实验5：多 Channel 卷积——一个输出 Channel 汇总所有输入 Channel
# ============================================================================
print_section("实验5：Multi-Channel Conv2d——输入 Channel 怎样被汇总")

multi_x = np.array([
    [
        [1, 2, 0, 1],
        [0, 1, 3, 1],
        [2, 1, 0, 0],
        [1, 0, 2, 2],
    ],
    [
        [0, 1, 1, 0],
        [2, 0, 1, 2],
        [1, 1, 0, 1],
        [0, 2, 1, 0],
    ],
], dtype=np.float32)  # (C_in=2,H=4,W=4)

multi_w = np.array([
    [
        [[1, 0, -1], [1, 0, -1], [1, 0, -1]],
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
    ],
    [
        [[1, 1, 1], [0, 0, 0], [-1, -1, -1]],
        [[1, 0, 1], [0, 1, 0], [1, 0, 1]],
    ],
], dtype=np.float32)  # (C_out=2,C_in=2,K=3,K=3)

multi_b = np.array([0.5, -0.25], dtype=np.float32)
manual_multi = conv2d_multi_channel_numpy(multi_x, multi_w, multi_b)
pt_multi = F.conv2d(
    torch.tensor(multi_x).unsqueeze(0),
    torch.tensor(multi_w),
    bias=torch.tensor(multi_b),
).squeeze(0).numpy()

print("input shape (C_in,H,W) =", multi_x.shape)
print("weight shape (C_out,C_in,K_h,K_w) =", multi_w.shape)
print("bias shape =", multi_b.shape)
print("output shape =", manual_multi.shape)
print("手写与 PyTorch 最大误差 =", float(np.max(np.abs(manual_multi - pt_multi))))
print("输出 Channel 0 =")
print(manual_multi[0])
print("输出 Channel 1 =")
print(manual_multi[1])


# ============================================================================
# 实验6：Max Pooling
# ============================================================================
print_section("实验6：Max Pooling——为什么空间尺寸会缩小")

pool_input = torch.tensor([
    [
        [1.0, 3.0, 2.0, 0.0],
        [4.0, 6.0, 1.0, 2.0],
        [0.0, 2.0, 8.0, 7.0],
        [1.0, 5.0, 3.0, 9.0],
    ]
]).unsqueeze(0)  # (1,1,4,4)

pooled = F.max_pool2d(pool_input, kernel_size=2, stride=2)
print("MaxPool 输入 shape =", tuple(pool_input.shape))
print(pool_input.squeeze().numpy())
print("MaxPool 输出 shape =", tuple(pooled.shape))
print(pooled.squeeze().numpy())

# 图3：Pooling 前后。
fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
for ax, data, title in zip(
    axes,
    [pool_input.squeeze().numpy(), pooled.squeeze().numpy()],
    ["Pooling 前 4×4", "MaxPool 后 2×2"],
):
    im = ax.imshow(data, cmap="magma")
    ax.set_title(title)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i,j]:.0f}", ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
fig.savefig(FIG_DIR / "lesson48_max_pooling.png", dpi=180)
plt.close(fig)


# ============================================================================
# 实验7：Translation Equivariance——输入平移，Feature Map 也平移
# ============================================================================
print_section("实验7：Translation Equivariance——局部模式移动后 Feature Map 怎样变化")

base = torch.zeros(1, 1, 9, 9)
base[0, 0, 4, 4] = 1.0
shifted = torch.zeros_like(base)
shifted[0, 0, 4, 5] = 1.0  # 向右移动 1 格

blur_kernel = torch.tensor([
    [0.0, 1.0, 0.0],
    [1.0, 2.0, 1.0],
    [0.0, 1.0, 0.0],
]).view(1, 1, 3, 3)

feat_base = F.conv2d(base, blur_kernel, padding=1)
feat_shifted = F.conv2d(shifted, blur_kernel, padding=1)

# 将原 Feature Map 向右平移一格，与 shifted 的输出比较。
feat_base_shifted = torch.zeros_like(feat_base)
feat_base_shifted[:, :, :, 1:] = feat_base[:, :, :, :-1]
interior_diff = (feat_base_shifted[:, :, :, 1:-1] - feat_shifted[:, :, :, 1:-1]).abs().max().item()

max_base = np.unravel_index(np.argmax(feat_base.squeeze().numpy()), (9, 9))
max_shift = np.unravel_index(np.argmax(feat_shifted.squeeze().numpy()), (9, 9))
print("原输入最大响应位置 =", max_base)
print("右移 1 格后最大响应位置 =", max_shift)
print("忽略边界后，平移后的 Feature Map 最大差异 =", interior_diff)
print("结论：卷积更接近 translation equivariance（输入移动，响应也移动），不是自动完全 translation invariant。")

# 图4：平移前后 Feature Map。
fig, axes = plt.subplots(2, 2, figsize=(7, 7))
for ax, data, title in [
    (axes[0,0], base.squeeze().numpy(), "原输入"),
    (axes[0,1], shifted.squeeze().numpy(), "输入右移 1 格"),
    (axes[1,0], feat_base.squeeze().numpy(), "原 Feature Map"),
    (axes[1,1], feat_shifted.squeeze().numpy(), "Feature Map 也右移"),
]:
    ax.imshow(data, cmap="viridis")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()
fig.savefig(FIG_DIR / "lesson48_translation_equivariance.png", dpi=180)
plt.close(fig)


# ============================================================================
# 实验8：TinyCNN——跟踪完整 Shape 与参数量
# ============================================================================
print_section("实验8：TinyCNN——Conv → ReLU → Pool → Flatten → Linear 的 Shape 流")

class TinyCNN(nn.Module):
    """一个只用于理解 Shape 的极小 CNN。"""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(4 * 4 * 4, 2)

    def forward(self, x):
        z1 = self.conv(x)
        a1 = self.relu(z1)
        p1 = self.pool(a1)
        flat = torch.flatten(p1, start_dim=1)
        y = self.fc(flat)
        return y, (z1, a1, p1, flat)


model = TinyCNN()
tiny_x = torch.randn(5, 1, 8, 8)
tiny_y, (z1, a1, p1, flat) = model(tiny_x)

print("输入 x shape =", tuple(tiny_x.shape))
print("Conv 输出 z1 shape =", tuple(z1.shape))
print("ReLU 输出 a1 shape =", tuple(a1.shape))
print("MaxPool 输出 p1 shape =", tuple(p1.shape))
print("Flatten 输出 shape =", tuple(flat.shape))
print("最终 logits shape =", tuple(tiny_y.shape))
print("Conv 参数量 =", sum(p.numel() for p in model.conv.parameters()))
print("FC 参数量 =", sum(p.numel() for p in model.fc.parameters()))
print("总参数量 =", sum(p.numel() for p in model.parameters()))

# 图5：Shape Pipeline。
fig = plt.figure(figsize=(11, 3.2))
ax = fig.add_subplot(111)
ax.axis("off")
labels = [
    "Input\n(5,1,8,8)",
    "Conv2d\n(5,4,8,8)",
    "ReLU\n(5,4,8,8)",
    "MaxPool\n(5,4,4,4)",
    "Flatten\n(5,64)",
    "Linear\n(5,2)",
]
xs = np.linspace(0.08, 0.92, len(labels))
for i, (xpos, label) in enumerate(zip(xs, labels)):
    ax.text(xpos, 0.5, label, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black"))
    if i < len(labels) - 1:
        ax.annotate("", xy=(xs[i+1]-0.07, 0.5), xytext=(xpos+0.07, 0.5), arrowprops=dict(arrowstyle="->"))
ax.set_title("TinyCNN：每一步都先追踪 Shape")
fig.tight_layout()
fig.savefig(FIG_DIR / "lesson48_tinycnn_shape_flow.png", dpi=180)
plt.close(fig)

print("\n全部实验完成。")
print("图像输出目录：", FIG_DIR)
