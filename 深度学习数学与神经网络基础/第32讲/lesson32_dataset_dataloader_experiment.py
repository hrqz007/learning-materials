# 第32讲实验：Dataset、DataLoader 与 Mini-Batch
# 目标：真正观察 Sample -> Dataset -> DataLoader -> Batch -> Training Step 的完整数据流。

from pathlib import Path
import sys
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from matplotlib import pyplot as plt
from matplotlib import font_manager

# ----------------------------------------------------------------------------
# 绘图与输出目录
# ----------------------------------------------------------------------------
CJK_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(CJK_FONT_PATH)
CJK_FONT = font_manager.FontProperties(fname=CJK_FONT_PATH)
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)


def title(text):
    """打印实验标题。"""
    print("\n" + "=" * 96)
    print(text)
    print("=" * 96)


# ----------------------------------------------------------------------------
# 实验0：环境信息
# ----------------------------------------------------------------------------
title("实验0：环境信息")
print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())


# ----------------------------------------------------------------------------
# 实验1：先不用 DataLoader，直接观察“一个 Sample”与“整个 Dataset”
# ----------------------------------------------------------------------------
title("实验1：Sample、Feature、Label 与 Dataset")

# 造 10 个最简单的回归样本：y = 3x - 2
X = torch.arange(10, dtype=torch.float32).reshape(-1, 1)
Y = 3.0 * X - 2.0

print("X.shape =", tuple(X.shape))
print("Y.shape =", tuple(Y.shape))
print("Dataset 样本数 N =", len(X))
print("第0个 sample: x =", X[0].tolist(), "y =", Y[0].tolist())
print("第5个 sample: x =", X[5].tolist(), "y =", Y[5].tolist())

# TensorDataset 把已经存在的 Tensor 组织成 Dataset
basic_dataset = TensorDataset(X, Y)
print("len(TensorDataset) =", len(basic_dataset))
x3, y3 = basic_dataset[3]
print("basic_dataset[3] -> x =", x3.tolist(), "y =", y3.tolist())


# ----------------------------------------------------------------------------
# 实验2：手工切 Mini-Batch，与 DataLoader 的输出对照
# ----------------------------------------------------------------------------
title("实验2：手工切 Mini-Batch vs DataLoader")

batch_size = 4
print("N =", len(basic_dataset), "batch_size =", batch_size)
print("理论 batches/epoch = ceil(N / batch_size) =", math.ceil(len(basic_dataset) / batch_size))

print("\n手工切片：")
manual_batches = []
for start in range(0, len(X), batch_size):
    xb = X[start:start + batch_size]
    yb = Y[start:start + batch_size]
    manual_batches.append((xb.clone(), yb.clone()))
    print(
        f"start={start:2d} | xb.shape={tuple(xb.shape)} | "
        f"x values={xb.flatten().tolist()}"
    )

loader_no_shuffle = DataLoader(
    basic_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
)

print("\nDataLoader(shuffle=False, drop_last=False)：")
loader_batches = []
for batch_idx, (xb, yb) in enumerate(loader_no_shuffle):
    loader_batches.append((xb.clone(), yb.clone()))
    print(
        f"batch_idx={batch_idx} | xb.shape={tuple(xb.shape)} | "
        f"x values={xb.flatten().tolist()}"
    )

# 检查手工切片与 DataLoader 完全一致
same = True
for (mx, my), (lx, ly) in zip(manual_batches, loader_batches):
    same = same and torch.equal(mx, lx) and torch.equal(my, ly)
print("手工切片与 DataLoader 完全一致:", same)


# ----------------------------------------------------------------------------
# 实验3：drop_last=False / True 到底改变什么？
# ----------------------------------------------------------------------------
title("实验3：drop_last 对最后一个不足 Batch 的影响")

loader_keep = DataLoader(basic_dataset, batch_size=4, shuffle=False, drop_last=False)
loader_drop = DataLoader(basic_dataset, batch_size=4, shuffle=False, drop_last=True)

keep_sizes = [len(xb) for xb, _ in loader_keep]
drop_sizes = [len(xb) for xb, _ in loader_drop]
print("drop_last=False batch sizes =", keep_sizes)
print("drop_last=True  batch sizes =", drop_sizes)
print("len(loader_keep) =", len(loader_keep))
print("len(loader_drop) =", len(loader_drop))


# ----------------------------------------------------------------------------
# 实验4：shuffle=True 只改变样本顺序，不改变样本内容
# ----------------------------------------------------------------------------
title("实验4：shuffle=True 到底做了什么？")

# 使用独立 Generator 固定 shuffle 的随机序列，使实验可复现
g1 = torch.Generator().manual_seed(3201)
g2 = torch.Generator().manual_seed(3202)
loader_shuffle_1 = DataLoader(basic_dataset, batch_size=4, shuffle=True, generator=g1)
loader_shuffle_2 = DataLoader(basic_dataset, batch_size=4, shuffle=True, generator=g2)

order1 = torch.cat([xb.flatten() for xb, _ in loader_shuffle_1]).tolist()
order2 = torch.cat([xb.flatten() for xb, _ in loader_shuffle_2]).tolist()
print("shuffle order #1 =", order1)
print("shuffle order #2 =", order2)
print("order #1 与原顺序相同:", order1 == list(range(10)))
print("order #1 排序后仍为 0..9:", sorted(order1) == [float(i) for i in range(10)])


# ----------------------------------------------------------------------------
# 实验5：自己写一个 Custom Dataset，观察 __len__ / __getitem__
# ----------------------------------------------------------------------------
title("实验5：自定义 Dataset —— __len__ 与 __getitem__")

class ToyRegressionDataset(Dataset):
    """最小自定义 Dataset：每个索引返回一对 (x, y)。"""

    def __init__(self, n=12):
        self.x = torch.linspace(-1.0, 1.0, steps=n).reshape(-1, 1)
        self.y = 3.0 * self.x - 2.0

    def __len__(self):
        # 告诉 DataLoader：一共有多少个 sample
        return len(self.x)

    def __getitem__(self, idx):
        # 告诉 DataLoader：第 idx 个 sample 到底是什么
        return self.x[idx], self.y[idx]


custom_dataset = ToyRegressionDataset(n=12)
print("len(custom_dataset) =", len(custom_dataset))
print("custom_dataset[0] =", tuple(v.tolist() for v in custom_dataset[0]))
print("custom_dataset[7] =", tuple(v.tolist() for v in custom_dataset[7]))

custom_loader = DataLoader(custom_dataset, batch_size=5, shuffle=False)
for batch_idx, (xb, yb) in enumerate(custom_loader):
    print(
        f"batch {batch_idx}: X.shape={tuple(xb.shape)}, Y.shape={tuple(yb.shape)}, "
        f"first_x={xb[0].item():.3f}"
    )


# ----------------------------------------------------------------------------
# 实验6：Epoch、Batch、Step 的精确关系
# ----------------------------------------------------------------------------
title("实验6：Epoch、Batch、Step 的精确关系")

N = 23
B = 6
num_epochs = 3
step_dataset = TensorDataset(
    torch.arange(N, dtype=torch.float32).reshape(-1, 1),
    torch.zeros(N, 1),
)
step_loader = DataLoader(step_dataset, batch_size=B, shuffle=False, drop_last=False)

print("N =", N, "batch_size =", B)
print("len(loader) =", len(step_loader))
print("ceil(N/B) =", math.ceil(N / B))
print("如果训练", num_epochs, "个 epoch，总 optimizer steps =", num_epochs * len(step_loader))

step_counter = 0
for epoch in range(num_epochs):
    batch_sizes = []
    for xb, yb in step_loader:
        batch_sizes.append(len(xb))
        step_counter += 1
    print(f"epoch={epoch} batch_sizes={batch_sizes} cumulative_steps={step_counter}")


# ----------------------------------------------------------------------------
# 实验7：把 DataLoader 正式接进 Training Loop
# ----------------------------------------------------------------------------
title("实验7：DataLoader + Training Loop —— Mini-Batch 训练")

torch.manual_seed(32)
train_x = torch.linspace(-2.0, 2.0, steps=80).reshape(-1, 1)
train_y = 3.0 * train_x - 2.0
train_dataset = TensorDataset(train_x, train_y)

# 为了可复现，每个实验都使用固定的 shuffle generator
train_generator = torch.Generator().manual_seed(3200)
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    generator=train_generator,
    drop_last=False,
)

model = nn.Linear(1, 1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

num_epochs = 20
loss_history = []
epoch_mean_losses = []
global_step = 0

for epoch in range(num_epochs):
    epoch_losses = []
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()

        global_step += 1
        loss_history.append(loss.item())
        epoch_losses.append(loss.item())

    epoch_mean = float(np.mean(epoch_losses))
    epoch_mean_losses.append(epoch_mean)
    if epoch in [0, 1, 2, 4, 9, 19]:
        print(
            f"epoch={epoch:2d} | mean_batch_loss={epoch_mean:.8f} | "
            f"weight={model.weight.item():.6f} | bias={model.bias.item():.6f} | "
            f"global_step={global_step}"
        )

print("len(train_loader) =", len(train_loader))
print("总 global_step =", global_step)
print("理论总 step = num_epochs * len(loader) =", num_epochs * len(train_loader))
print("最终 weight =", model.weight.item())
print("最终 bias   =", model.bias.item())
print("最终 epoch mean loss =", epoch_mean_losses[-1])

# 保存 Step Loss 曲线
plt.figure(figsize=(7.2, 4.6))
plt.plot(np.arange(1, len(loss_history) + 1), loss_history)
plt.xlabel("Global Step", fontproperties=CJK_FONT)
plt.ylabel("Mini-batch MSE Loss", fontproperties=CJK_FONT)
plt.title("Mini-Batch Training：每个 Step 的 Loss", fontproperties=CJK_FONT)
plt.tight_layout()
plt.savefig(FIG_DIR / "lesson32_minibatch_training_loss.png", dpi=180)
plt.close()

# 保存 Epoch 平均 Loss 曲线
plt.figure(figsize=(7.2, 4.6))
plt.plot(np.arange(1, num_epochs + 1), epoch_mean_losses, marker="o")
plt.xlabel("Epoch", fontproperties=CJK_FONT)
plt.ylabel("Mean Batch Loss", fontproperties=CJK_FONT)
plt.title("每个 Epoch 的平均 Mini-Batch Loss", fontproperties=CJK_FONT)
plt.tight_layout()
plt.savefig(FIG_DIR / "lesson32_epoch_mean_loss.png", dpi=180)
plt.close()


# ----------------------------------------------------------------------------
# 实验8：画出 Batch 划分与 Shuffle 的直观示意图
# ----------------------------------------------------------------------------
title("实验8：生成 Batch / Shuffle 示意图")

# 图1：N=10, B=4 的 batch 划分
fig, ax = plt.subplots(figsize=(9.2, 2.5))
for i in range(10):
    if i < 4:
        row = 0
        group = "Batch 0"
    elif i < 8:
        row = 0
        group = "Batch 1"
    else:
        row = 0
        group = "Batch 2"
    ax.add_patch(plt.Rectangle((i, 0), 0.9, 0.8, fill=False, linewidth=1.5))
    ax.text(i + 0.45, 0.4, str(i), ha="center", va="center", fontsize=11)
ax.text(1.95, 1.05, "Batch 0: 4 个样本", ha="center", fontproperties=CJK_FONT)
ax.text(5.95, 1.05, "Batch 1: 4 个样本", ha="center", fontproperties=CJK_FONT)
ax.text(8.95, 1.05, "Batch 2: 2 个样本", ha="center", fontproperties=CJK_FONT)
ax.set_xlim(-0.1, 10)
ax.set_ylim(-0.1, 1.45)
ax.axis("off")
ax.set_title("N=10, batch_size=4, drop_last=False", fontproperties=CJK_FONT)
plt.tight_layout()
plt.savefig(FIG_DIR / "lesson32_batch_partition.png", dpi=180)
plt.close()

# 图2：shuffle 前后顺序
original = np.arange(10)
shuffled = np.array(order1, dtype=int)
fig, axes = plt.subplots(2, 1, figsize=(9.2, 3.8))
for ax, values, label in zip(axes, [original, shuffled], ["原始顺序", "shuffle 后顺序"]):
    for i, v in enumerate(values):
        ax.add_patch(plt.Rectangle((i, 0), 0.9, 0.75, fill=False, linewidth=1.3))
        ax.text(i + 0.45, 0.375, str(v), ha="center", va="center", fontsize=10)
    ax.set_xlim(-0.1, 10)
    ax.set_ylim(-0.1, 1.0)
    ax.axis("off")
    ax.text(-0.5, 0.38, label, ha="right", va="center", fontproperties=CJK_FONT)
plt.suptitle("Shuffle 改变样本访问顺序，但样本集合没有改变", fontproperties=CJK_FONT)
plt.tight_layout()
plt.savefig(FIG_DIR / "lesson32_shuffle_order.png", dpi=180)
plt.close()

# 图3：Dataset -> DataLoader -> Batch -> Training Step 流程
fig, ax = plt.subplots(figsize=(10, 3.1))
ax.axis("off")
boxes = [
    (0.03, 0.35, 0.16, 0.35, "Dataset\n单个 Sample"),
    (0.25, 0.35, 0.18, 0.35, "DataLoader\n取样 + 组 Batch"),
    (0.49, 0.35, 0.16, 0.35, "Mini-Batch\nXb, Yb"),
    (0.71, 0.35, 0.23, 0.35, "Training Step\nForward/Loss/Backward/Step"),
]
for x0, y0, w, h, txt in boxes:
    ax.add_patch(plt.Rectangle((x0, y0), w, h, fill=False, linewidth=1.6))
    ax.text(x0 + w / 2, y0 + h / 2, txt, ha="center", va="center", fontproperties=CJK_FONT, fontsize=11)
for x_start, x_end in [(0.19, 0.25), (0.43, 0.49), (0.65, 0.71)]:
    ax.annotate("", xy=(x_end, 0.525), xytext=(x_start, 0.525), arrowprops=dict(arrowstyle="->", lw=1.6))
ax.text(0.5, 0.12, "一个 Epoch = DataLoader 被完整遍历一遍", ha="center", fontproperties=CJK_FONT, fontsize=11)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(FIG_DIR / "lesson32_data_pipeline.png", dpi=180)
plt.close()

print("图像已保存到:", FIG_DIR)
print("实验完成。")
