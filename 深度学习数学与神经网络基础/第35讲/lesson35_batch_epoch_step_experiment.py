# -*- coding: utf-8 -*-
"""
第35讲实验：Batch、Epoch、Step 三个训练时间尺度

目标：
1. 用 DataLoader 实测每个 Epoch 的 Batch 数量；
2. 对比 drop_last=False / True；
3. 区分 micro-batch iteration 与 optimizer step；
4. 观察 gradient accumulation 在 Epoch 边界处的计数差异；
5. 计算多 GPU + gradient accumulation 下的 global effective batch；
6. 演示 scheduler 如果跟错“step”会发生什么；
7. 用 LLM 风格 token 数量重新理解训练预算。
"""

# 导入数学库，用于向上取整等计算。
import math
# 导入 pathlib，用于可靠地创建结果目录。
from pathlib import Path
# 导入 NumPy，用于构造数组和简单数值实验。
import numpy as np
# 导入 PyTorch。
import torch
# 导入 PyTorch 的 Dataset 与 DataLoader。
from torch.utils.data import TensorDataset, DataLoader
# 导入 Matplotlib，用于保存实验图。
import matplotlib.pyplot as plt
# 导入字体管理模块，用于给图中的中文选择可用字体。
from matplotlib import font_manager

# 固定随机种子，保证重复运行时结果一致。
torch.manual_seed(35)
# 同时固定 NumPy 的随机种子。
np.random.seed(35)

# 找到当前脚本所在目录。
ROOT = Path(__file__).resolve().parent
# 创建图片输出目录。
FIG_DIR = ROOT / "figures"
# 如果目录已经存在，则不报错。
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 尝试找到系统中的中文字体，避免图中中文显示为方块。
font_candidates = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
]
# 逐个检查字体是否存在。
for font_path in font_candidates:
    # 如果当前候选字体存在，就注册并使用它。
    if Path(font_path).exists():
        # 把字体文件注册给 Matplotlib。
        font_manager.fontManager.addfont(font_path)
        # 读取字体的真实名称。
        font_name = font_manager.FontProperties(fname=font_path).get_name()
        # 设置为默认无衬线字体。
        plt.rcParams["font.sans-serif"] = [font_name]
        # 找到一个可用字体以后就退出循环。
        break
# 让坐标轴负号正常显示。
plt.rcParams["axes.unicode_minus"] = False


# 定义一个打印分隔标题的小函数。
def section(title: str) -> None:
    # 打印空行，增强可读性。
    print()
    # 打印分隔线。
    print("=" * 78)
    # 打印实验标题。
    print(title)
    # 再打印分隔线。
    print("=" * 78)


# 定义一个根据数据量和 Batch Size 计算每个 Epoch Batch 数量的函数。
def batches_per_epoch(n: int, batch_size: int, drop_last: bool) -> int:
    # drop_last=True 时，只保留完整 Batch，因此使用整除。
    if drop_last:
        # 返回完整 Batch 数量。
        return n // batch_size
    # drop_last=False 时，最后不足一个 Batch 的样本也会形成小 Batch，因此向上取整。
    return math.ceil(n / batch_size)


# 实验1：真实 DataLoader 的 Batch 数量与理论公式是否一致。
section("实验1：DataLoader 的 Batch 数量、最后一个 Batch 与 drop_last")
# 设置样本总数。
N = 23
# 设置 Batch Size。
B = 6
# 构造 23 个简单的一维样本。
x = torch.arange(N, dtype=torch.float32).reshape(-1, 1)
# 构造对应标签，这里标签本身不重要，只为了组成 Dataset。
y = 2.0 * x + 1.0
# 把输入和标签封装成 TensorDataset。
dataset = TensorDataset(x, y)

# 遍历两种 drop_last 设置。
for drop_last in [False, True]:
    # 创建 DataLoader，并关闭 shuffle，方便直接观察样本编号。
    loader = DataLoader(dataset, batch_size=B, shuffle=False, drop_last=drop_last)
    # 用列表记录每个 Batch 的大小。
    batch_sizes = []
    # 用列表记录每个 Batch 中的第一个和最后一个样本编号。
    batch_ranges = []
    # 遍历一个 Epoch 中的所有 Batch。
    for xb, yb in loader:
        # 记录当前 Batch 的样本数量。
        batch_sizes.append(len(xb))
        # 记录当前 Batch 的样本编号范围。
        batch_ranges.append((int(xb[0].item()), int(xb[-1].item())))
    # 用理论公式计算 Batch 数。
    expected = batches_per_epoch(N, B, drop_last)
    # 打印设置。
    print(f"drop_last={drop_last}")
    # 打印 DataLoader 实际 Batch 数。
    print(f"  实际 batches/epoch = {len(loader)}")
    # 打印理论 Batch 数。
    print(f"  理论 batches/epoch = {expected}")
    # 打印每个 Batch 的大小。
    print(f"  batch sizes        = {batch_sizes}")
    # 打印每个 Batch 的样本范围。
    print(f"  sample ranges      = {batch_ranges}")

# 绘制 Batch 划分示意图。
fig, axes = plt.subplots(2, 1, figsize=(10, 4.8), sharex=True)
# 遍历两种设置并分别画图。
for ax, drop_last in zip(axes, [False, True]):
    # 计算当前设置下的 Batch 数。
    num_batches = batches_per_epoch(N, B, drop_last)
    # 从第一个 Batch 开始绘制。
    for batch_idx in range(num_batches):
        # 当前 Batch 的起始样本下标。
        start = batch_idx * B
        # 当前 Batch 的结束下标，不能超过 N。
        end = min(start + B, N)
        # drop_last=True 且最后不足完整 Batch 时，不绘制该 Batch。
        if drop_last and end - start < B:
            continue
        # 在横轴上画一个条块，表示当前 Batch 覆盖的样本区间。
        ax.barh(0, end - start, left=start, height=0.45, label=f"Batch {batch_idx+1}")
        # 在条块中间写 Batch 编号。
        ax.text((start + end) / 2, 0, f"B{batch_idx+1}\n{end-start} samples", ha="center", va="center", fontsize=9)
    # 设置纵轴标签。
    ax.set_yticks([])
    # 设置标题。
    ax.set_title(f"drop_last={drop_last}")
    # 设置横轴范围。
    ax.set_xlim(0, N)
# 设置横轴标签。
axes[-1].set_xlabel("样本索引位置")
# 设置总标题。
fig.suptitle("N=23, batch_size=6：一个 Epoch 怎样被切成 Batch")
# 调整布局，避免标题重叠。
fig.tight_layout(rect=[0, 0, 1, 0.95])
# 保存图片。
fig.savefig(FIG_DIR / "lesson35_batch_partition_drop_last.png", dpi=180)
# 关闭图形，释放内存。
plt.close(fig)


# 实验2：区分 micro-batch iteration 与 optimizer step。
section("实验2：Gradient Accumulation 后，micro-batch iteration 与 optimizer step 不再相等")
# 设置每个 Epoch 的 micro-batch 数量。
M = 7
# 设置梯度累积步数。
A = 3
# 计算如果每个 Epoch 结束时把剩余梯度也更新一次，需要多少 optimizer steps。
steps_flush_each_epoch = math.ceil(M / A)
# 计算如果直接丢弃不足 A 个 micro-batch 的剩余梯度，需要多少 optimizer steps。
steps_drop_remainder = M // A
# 打印结果。
print(f"micro-batches/epoch = {M}")
print(f"gradient_accumulation_steps = {A}")
print(f"每个 Epoch 结束时 flush 剩余梯度 -> optimizer steps/epoch = {steps_flush_each_epoch}")
print(f"若错误地丢弃不足 A 个的剩余梯度 -> optimizer steps/epoch = {steps_drop_remainder}")

# 模拟两个 Epoch，并记录每个 micro-batch 对应的 global optimizer step。
EPOCHS = 2
# 准备列表记录时间线。
timeline_micro = []
# 准备列表记录 optimizer step 编号。
timeline_opt = []
# 全局 optimizer step 从 0 开始。
global_opt_step = 0
# 遍历 Epoch。
for epoch in range(EPOCHS):
    # 当前累积了多少个 micro-batch。
    accum_count = 0
    # 遍历当前 Epoch 的 micro-batch。
    for micro_idx in range(M):
        # 累积计数加 1。
        accum_count += 1
        # 计算当前是不是应该做一次 optimizer.step()。
        should_step = (accum_count == A) or (micro_idx == M - 1)
        # 如果需要更新参数。
        if should_step:
            # optimizer step 编号加 1。
            global_opt_step += 1
            # 当前累积组清零。
            accum_count = 0
        # 记录当前 micro-batch 的全局序号。
        timeline_micro.append(epoch * M + micro_idx + 1)
        # 记录处理完当前 micro-batch 后已经发生了多少次 optimizer update。
        timeline_opt.append(global_opt_step)
# 打印两个 Epoch 总 micro-batch 数。
print(f"2 Epoch 总 micro-batch iterations = {EPOCHS * M}")
# 打印两个 Epoch 总 optimizer step 数。
print(f"2 Epoch 总 optimizer steps（每 Epoch flush） = {global_opt_step}")

# 绘制 micro-batch iteration 与 optimizer step 的关系。
fig, ax = plt.subplots(figsize=(9, 4.5))
# 用阶梯图展示 optimizer step 只在部分 micro-batch 后增加。
ax.step(timeline_micro, timeline_opt, where="post", marker="o")
# 设置横轴标题。
ax.set_xlabel("micro-batch iteration")
# 设置纵轴标题。
ax.set_ylabel("累计 optimizer step")
# 设置图标题。
ax.set_title("Gradient Accumulation：不是每个 micro-batch 都更新参数")
# 打开网格方便观察。
ax.grid(True, alpha=0.3)
# 调整布局。
fig.tight_layout()
# 保存图片。
fig.savefig(FIG_DIR / "lesson35_microbatch_vs_optimizer_step.png", dpi=180)
# 关闭图形。
plt.close(fig)


# 实验3：Epoch 边界是否 flush，会改变 optimizer step 的总计数。
section("实验3：Gradient Accumulation 跨 Epoch 还是每个 Epoch flush？")
# 设置每个 Epoch 7 个 micro-batch。
M = 7
# 设置累计 3 个 micro-batch 后更新一次。
A = 3
# 设置训练 2 个 Epoch。
E = 2
# 每个 Epoch 都 flush 时，总更新次数等于每个 Epoch 向上取整后相加。
flush_total = E * math.ceil(M / A)
# 如果完全跨 Epoch 连续累计，则总更新次数是总 micro-batch 数再向上取整。
continuous_total = math.ceil((E * M) / A)
# 打印两个结果。
print(f"每 Epoch flush: {flush_total} optimizer steps")
print(f"跨 Epoch 连续累计: {continuous_total} optimizer steps")
# 打印差异来源。
print("差异来自 Epoch 边界处不足 accumulation_steps 的残余 micro-batch 是否立即更新。")

# 绘制两种计数策略的柱状图。
fig, ax = plt.subplots(figsize=(7.5, 4.5))
# 画两个柱子。
ax.bar(["每 Epoch flush", "跨 Epoch 连续累计"], [flush_total, continuous_total])
# 设置纵轴标题。
ax.set_ylabel("2 Epoch 的 optimizer steps")
# 设置图标题。
ax.set_title("同样的 micro-batch 数，Epoch 边界策略会影响更新次数")
# 在柱子上标出数值。
for i, v in enumerate([flush_total, continuous_total]):
    ax.text(i, v + 0.08, str(v), ha="center", va="bottom")
# 调整布局。
fig.tight_layout()
# 保存图片。
fig.savefig(FIG_DIR / "lesson35_epoch_boundary_accumulation.png", dpi=180)
# 关闭图形。
plt.close(fig)


# 实验4：计算多 GPU + Gradient Accumulation 下的 Global Effective Batch。
section("实验4：Micro Batch、Global Batch、Effective Batch 与 Tokens per Update")
# 设置每张 GPU 一次能放的样本数。
micro_batch_per_device = 2
# 设置 GPU 数量。
world_size = 4
# 设置梯度累积次数。
grad_accum = 8
# 设置固定序列长度。
seq_len = 2048
# 计算一次 optimizer update 实际聚合了多少条序列。
global_effective_batch = micro_batch_per_device * world_size * grad_accum
# 计算每次 update 对应的 token slot 数量。
token_slots_per_update = global_effective_batch * seq_len
# 打印结果。
print(f"micro_batch_per_device = {micro_batch_per_device}")
print(f"world_size = {world_size}")
print(f"gradient_accumulation_steps = {grad_accum}")
print(f"global effective batch = {global_effective_batch} sequences/update")
print(f"sequence length = {seq_len}")
print(f"raw token slots/update = {token_slots_per_update:,}")

# 构造一个有 padding 的例子，说明 raw token slots 不等于有效 token。
valid_lengths = np.array([2048, 1900, 1700, 1200, 2048, 1500, 800, 2048], dtype=np.int64)
# 计算这 8 条序列的 raw token slot 数。
raw_slots_demo = len(valid_lengths) * seq_len
# 计算有效 token 总数。
valid_tokens_demo = int(valid_lengths.sum())
# 计算 padding token 数量。
padding_tokens_demo = raw_slots_demo - valid_tokens_demo
# 打印三种数量。
print(f"8 条示例序列的 raw token slots = {raw_slots_demo:,}")
print(f"8 条示例序列的 valid tokens    = {valid_tokens_demo:,}")
print(f"8 条示例序列的 padding tokens  = {padding_tokens_demo:,}")

# 绘制 raw token slots 与 valid token 的区别。
fig, ax = plt.subplots(figsize=(8, 4.5))
# 画三个柱子。
values = [raw_slots_demo, valid_tokens_demo, padding_tokens_demo]
labels = ["raw token slots", "valid tokens", "padding tokens"]
ax.bar(labels, values)
# 设置纵轴标题。
ax.set_ylabel("Token 数量")
# 设置图标题。
ax.set_title("LLM 训练：sequence batch 不等于有效 token 数")
# 给每个柱子写数值。
for i, v in enumerate(values):
    ax.text(i, v + 200, f"{v:,}", ha="center", va="bottom", fontsize=9)
# 调整布局。
fig.tight_layout()
# 保存图片。
fig.savefig(FIG_DIR / "lesson35_tokens_per_update.png", dpi=180)
# 关闭图形。
plt.close(fig)


# 实验5：scheduler 如果按 micro-batch 调用，会比 optimizer step 快很多。
section("实验5：Learning Rate Scheduler 应该跟哪个 step？")
# 设置总 micro-batch 数量。
total_micro = 20
# 设置梯度累积步数。
A = 4
# 设置初始学习率。
initial_lr = 0.1
# 设置每次 scheduler.step() 后的衰减比例。
gamma = 0.9
# 创建记录正确策略学习率的列表。
lr_correct = []
# 创建记录错误策略学习率的列表。
lr_wrong = []
# 正确策略的当前学习率。
lr_c = initial_lr
# 错误策略的当前学习率。
lr_w = initial_lr
# 记录已经发生的 optimizer step 数。
opt_count = 0
# 遍历所有 micro-batch。
for micro in range(1, total_micro + 1):
    # 错误策略：每个 micro-batch 都调用 scheduler.step()。
    lr_w *= gamma
    # 判断当前是否完成一个 accumulation group。
    if micro % A == 0:
        # optimizer 真正更新一次。
        opt_count += 1
        # 正确策略：只有 optimizer update 后 scheduler 才走一步。
        lr_c *= gamma
    # 保存正确策略当前学习率。
    lr_correct.append(lr_c)
    # 保存错误策略当前学习率。
    lr_wrong.append(lr_w)
# 打印 optimizer step 总数。
print(f"total micro-batches = {total_micro}")
print(f"gradient accumulation = {A}")
print(f"optimizer steps = {opt_count}")
# 打印两种策略训练结束时的学习率。
print(f"scheduler 按 optimizer step: final lr = {lr_correct[-1]:.8f}")
print(f"scheduler 按 micro-batch:    final lr = {lr_wrong[-1]:.8f}")

# 绘制两个学习率轨迹。
fig, ax = plt.subplots(figsize=(9, 4.8))
# 画正确策略。
ax.plot(range(1, total_micro + 1), lr_correct, marker="o", label="按 optimizer step 调 scheduler")
# 画错误策略。
ax.plot(range(1, total_micro + 1), lr_wrong, marker="x", label="每个 micro-batch 都调 scheduler")
# 设置横轴。
ax.set_xlabel("micro-batch iteration")
# 设置纵轴。
ax.set_ylabel("learning rate")
# 设置标题。
ax.set_title("Gradient Accumulation 下：scheduler 的“step”必须说清楚")
# 显示图例。
ax.legend()
# 打开网格。
ax.grid(True, alpha=0.3)
# 调整布局。
fig.tight_layout()
# 保存图片。
fig.savefig(FIG_DIR / "lesson35_scheduler_step_mismatch.png", dpi=180)
# 关闭图形。
plt.close(fig)


# 实验6：把 Epoch、micro-batch、optimizer step、sample 与 token 统一到一个例子里。
section("实验6：把 Epoch、Batch、Optimizer Step、Samples、Tokens 统一起来")
# 设置数据集样本数。
N = 1000
# 设置每张设备一次处理的序列数，这里先按单设备理解。
B = 128
# 设置 Epoch 数。
E = 10
# 设置是否丢弃最后不足完整 Batch 的样本。
drop_last = False
# 设置 Gradient Accumulation 次数。
A = 4
# 计算每个 Epoch 的 micro-batch 数。
M = batches_per_epoch(N, B, drop_last)
# 计算每个 Epoch 的 optimizer step 数，假设 Epoch 末尾 flush。
S = math.ceil(M / A)
# 计算总 micro-batch iteration 数。
total_micro = E * M
# 计算总 optimizer step 数。
total_opt = E * S
# 计算理论上每个 Epoch 被 DataLoader 取出的样本总数。
seen_samples_per_epoch = N
# 计算训练过程中的样本访问次数。
total_sample_visits = E * seen_samples_per_epoch
# 设置固定序列长度，用于 LLM 风格计算。
T = 512
# 计算 raw token slot 访问次数。
total_token_slots = total_sample_visits * T
# 打印各个尺度。
print(f"N={N}, batch_size={B}, epochs={E}, accumulation={A}, drop_last={drop_last}")
print(f"micro-batches/epoch = {M}")
print(f"optimizer steps/epoch（Epoch 末 flush） = {S}")
print(f"total micro-batch iterations = {total_micro}")
print(f"total optimizer steps = {total_opt}")
print(f"total sample visits = {total_sample_visits:,}")
print(f"若每条序列固定 {T} token，则 raw token slots = {total_token_slots:,}")

# 打印提醒信息，强调 step 这个词本身有歧义。
print()
print("结论提醒：日志里的 'step' 必须确认它到底指 micro-batch iteration、optimizer step、scheduler step 还是 global step。")
print("实验完成，图片已保存到 figures/ 目录。")
