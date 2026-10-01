# 第18讲实验：Cross Entropy 与 Next-token Prediction
# 目标：不用黑盒，亲手验证 Softmax、-log(p_correct)、Batch Cross Entropy 和 LLM token loss 的关系。

# 导入 NumPy，用来完成手工可解释的数组运算。
import numpy as np
# 导入 matplotlib，用来画“正确类别 logit 与 Cross Entropy”的关系图。
import matplotlib.pyplot as plt
# 尝试导入 PyTorch，用来和 NumPy 手写结果进行核对。
try:
    # 导入 torch 张量库。
    import torch
    # 导入 PyTorch 常用函数接口。
    import torch.nn.functional as F
    # 记录当前环境是否存在 PyTorch。
    TORCH_AVAILABLE = True
except Exception:
    # 如果环境中没有 PyTorch，就跳过 PyTorch 对照实验。
    TORCH_AVAILABLE = False

# 设置 NumPy 打印格式，减少无关小数位，便于初学者阅读。
np.set_printoptions(precision=6, suppress=True)

# 定义稳定版 Softmax：先减去最大值，再做指数和归一化。
def stable_softmax(logits, axis=-1):
    # 在指定维度上找到最大 logit，并保留维度，方便后续广播。
    max_logits = np.max(logits, axis=axis, keepdims=True)
    # 所有 logits 同时减去同一个最大值，不改变 Softmax 最终结果。
    shifted = logits - max_logits
    # 对平移后的 logits 逐元素取指数，使所有数都变成正数。
    exp_values = np.exp(shifted)
    # 在类别维度上求和，得到归一化分母。
    denominator = np.sum(exp_values, axis=axis, keepdims=True)
    # 每个指数值除以总和，得到概率分布。
    probabilities = exp_values / denominator
    # 返回概率。
    return probabilities

# 定义单样本 Cross Entropy：输入 logits 和正确类别索引。
def cross_entropy_single(logits, target_index):
    # 先把 logits 转成概率。
    probabilities = stable_softmax(logits)
    # 取出正确类别对应的概率。
    p_correct = probabilities[target_index]
    # 对正确类别概率取负对数，得到 Cross Entropy。
    loss = -np.log(p_correct)
    # 同时返回 loss 和完整概率，方便观察。
    return loss, probabilities

# 定义 Batch Cross Entropy：每个样本都有一组 logits 和一个 target index。
def cross_entropy_batch(logits_batch, targets):
    # 对每个样本在最后一个类别维度上做 Softmax。
    probabilities = stable_softmax(logits_batch, axis=-1)
    # 生成样本索引 0,1,2,...,B-1。
    batch_indices = np.arange(logits_batch.shape[0])
    # 从每一行中取出该样本正确类别的概率。
    p_correct = probabilities[batch_indices, targets]
    # 对每个正确类别概率取负对数，得到每个样本的 loss。
    per_sample_loss = -np.log(p_correct)
    # 对样本 loss 求平均，得到常见的 mean reduction。
    mean_loss = np.mean(per_sample_loss)
    # 返回每个样本 loss、平均 loss 和概率矩阵。
    return per_sample_loss, mean_loss, probabilities

# 打印分隔线，便于阅读实验输出。
def section(title):
    # 打印空行和横线。
    print("\n" + "=" * 72)
    # 打印当前实验标题。
    print(title)
    # 再打印横线。
    print("=" * 72)

# 实验1：单样本，手工观察 Softmax -> p_correct -> -log(p_correct)。
section("实验1：单样本 Cross Entropy 的完整链条")
# 定义三个类别的 logits。
logits = np.array([2.0, 1.0, 0.1], dtype=np.float64)
# 假设类别0是真实正确答案。
target = 0
# 调用手写 Cross Entropy。
loss, probabilities = cross_entropy_single(logits, target)
# 打印原始 logits。
print("logits =", logits)
# 打印 Softmax 概率。
print("softmax probabilities =", probabilities)
# 打印概率和，验证它们组成合法概率分布。
print("probability sum =", probabilities.sum())
# 打印正确类别概率。
print("p_correct =", probabilities[target])
# 打印最终 Cross Entropy。
print("cross entropy = -log(p_correct) =", loss)

# 实验2：同一组 logits，只改变“哪个类别是正确答案”。
section("实验2：同一组 logits，不同 target 会得到不同 loss")
# 依次把 0、1、2 当作正确类别。
for current_target in [0, 1, 2]:
    # 计算当前 target 对应的 Cross Entropy。
    current_loss, current_probabilities = cross_entropy_single(logits, current_target)
    # 打印 target、正确类别概率和 loss。
    print(
        f"target={current_target}, "
        f"p_correct={current_probabilities[current_target]:.6f}, "
        f"loss={current_loss:.6f}"
    )

# 实验3：把正确类别 logit 不断提高，观察 loss 是否下降。
section("实验3：提高正确类别 logit，Cross Entropy 是否下降")
# 固定另外两个类别的 logits，只改变正确类别 logit。
correct_logits = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 5.0])
# 创建空列表，保存每个位置的正确类别概率。
correct_probabilities = []
# 创建空列表，保存每个位置的 loss。
losses = []
# 遍历不同的正确类别 logit。
for correct_logit in correct_logits:
    # 构造三分类 logits，其中类别0是真实答案。
    current_logits = np.array([correct_logit, 1.0, 0.1], dtype=np.float64)
    # 计算 loss 和概率。
    current_loss, current_prob = cross_entropy_single(current_logits, 0)
    # 保存正确类别概率。
    correct_probabilities.append(current_prob[0])
    # 保存 loss。
    losses.append(current_loss)
    # 打印当前结果。
    print(
        f"correct_logit={correct_logit:>4.1f}, "
        f"p_correct={current_prob[0]:.6f}, "
        f"loss={current_loss:.6f}"
    )
# 把列表转换成 NumPy 数组，方便绘图。
correct_probabilities = np.array(correct_probabilities)
# 把 loss 列表转换成 NumPy 数组。
losses = np.array(losses)

# 实验4：验证 one-hot Cross Entropy 与 -log(p_correct) 等价。
section("实验4：one-hot 公式为什么会退化成 -log(p_correct)")
# 定义 one-hot target：类别0正确，所以第0项为1，其余为0。
one_hot = np.array([1.0, 0.0, 0.0])
# 按完整公式计算：-sum(y_i * log(p_i))。
one_hot_loss = -np.sum(one_hot * np.log(probabilities))
# 打印 one-hot target。
print("one_hot target =", one_hot)
# 打印完整 one-hot Cross Entropy。
print("-sum(y * log(p)) =", one_hot_loss)
# 打印简化后的 -log(p_correct)。
print("-log(p_correct) =", -np.log(probabilities[target]))
# 打印两者差值，理论上应接近0。
print("difference =", abs(one_hot_loss - (-np.log(probabilities[target]))))

# 实验5：Batch Cross Entropy。
section("实验5：Batch Cross Entropy 与 mean reduction")
# 构造3个样本，每个样本有4个类别的 logits。
logits_batch = np.array(
    [
        [3.0, 1.0, 0.0, -1.0],
        [0.2, 0.1, 2.4, 0.0],
        [0.0, 1.8, 0.5, -0.2],
    ],
    dtype=np.float64,
)
# 定义每个样本的正确类别索引。
targets = np.array([0, 2, 1], dtype=np.int64)
# 计算 Batch Cross Entropy。
per_sample_loss, mean_loss, batch_probabilities = cross_entropy_batch(logits_batch, targets)
# 打印 logits Shape。
print("logits_batch.shape =", logits_batch.shape)
# 打印 targets Shape。
print("targets.shape =", targets.shape)
# 打印概率矩阵。
print("probabilities =\n", batch_probabilities)
# 打印每个样本的 loss。
print("per_sample_loss =", per_sample_loss)
# 打印平均 loss。
print("mean_loss =", mean_loss)

# 实验6：数值稳定性。大 logits 下 naive 写法可能溢出。
section("实验6：Cross Entropy 的数值稳定性")
# 构造很大的 logits。
large_logits = np.array([1000.0, 1001.0, 999.0], dtype=np.float64)
# 设置类别1为正确答案。
large_target = 1
# 关闭 NumPy 的溢出警告，避免输出太杂乱；我们会主动打印结果。
with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
    # naive 做法：直接对原始 logits 取指数。
    naive_exp = np.exp(large_logits)
    # naive 做法：指数后归一化。
    naive_prob = naive_exp / np.sum(naive_exp)
    # naive 做法：取正确类别概率的负对数。
    naive_loss = -np.log(naive_prob[large_target])
# 使用稳定版 Softmax + Cross Entropy。
stable_loss, stable_prob = cross_entropy_single(large_logits, large_target)
# 打印 naive 指数结果。
print("naive exp =", naive_exp)
# 打印 naive 概率结果。
print("naive probabilities =", naive_prob)
# 打印 naive loss。
print("naive loss =", naive_loss)
# 打印稳定版概率。
print("stable probabilities =", stable_prob)
# 打印稳定版 loss。
print("stable loss =", stable_loss)

# 实验7：PyTorch CrossEntropyLoss 直接接收 logits，而不是 Softmax 后的概率。
section("实验7：NumPy 手写 Cross Entropy 与 PyTorch 对照")
# 只有环境中存在 PyTorch 才执行这一段。
if TORCH_AVAILABLE:
    # 把 NumPy logits 转成 float64 的 PyTorch Tensor，提高对照精度。
    torch_logits = torch.tensor(logits_batch, dtype=torch.float64)
    # 把 target 索引转成 PyTorch long Tensor。
    torch_targets = torch.tensor(targets, dtype=torch.long)
    # 使用 reduction='none' 获得每个样本各自的 Cross Entropy。
    torch_per_sample = F.cross_entropy(torch_logits, torch_targets, reduction="none")
    # 使用默认 mean reduction 获得平均 loss。
    torch_mean = F.cross_entropy(torch_logits, torch_targets, reduction="mean")
    # 打印 PyTorch 每样本 loss。
    print("torch per_sample_loss =", torch_per_sample.detach().numpy())
    # 打印 PyTorch 平均 loss。
    print("torch mean_loss =", torch_mean.item())
    # 计算 NumPy 与 PyTorch 的最大差异。
    max_diff = np.max(np.abs(torch_per_sample.detach().numpy() - per_sample_loss))
    # 打印最大差异。
    print("NumPy vs PyTorch max abs diff =", max_diff)
    # 演示一个常见错误：先 Softmax 后再传给 F.cross_entropy。
    wrong_input = torch.softmax(torch_logits, dim=-1)
    # 这里不会报语法错误，但数学含义已经错了，因为 F.cross_entropy 期望 logits。
    wrong_loss = F.cross_entropy(wrong_input, torch_targets, reduction="mean")
    # 打印错误用法得到的值，用于和正确 loss 对照。
    print("WRONG: cross_entropy(softmax(logits), target) =", wrong_loss.item())
    # 打印正确用法。
    print("RIGHT: cross_entropy(logits, target) =", torch_mean.item())
else:
    # 如果没有 PyTorch，给出说明。
    print("当前环境未安装 PyTorch，已跳过 PyTorch 对照实验。")

# 实验8：微型 LLM Next-token Prediction Loss。
section("实验8：微型 LLM 的 Next-token Prediction Loss")
# 定义一个玩具词表：A、B、C、D、E 共5个 token。
vocab = ["A", "B", "C", "D", "E"]
# 假设原始 token 序列是 A -> B -> C -> D -> E。
sequence = np.array([0, 1, 2, 3, 4], dtype=np.int64)
# 模型在前4个位置输出对5个词表 token 的 logits，所以 Shape 是 (T=4, V=5)。
lm_logits = np.array(
    [
        [0.2, 2.2, 0.0, -1.0, -0.5],
        [-0.2, 0.1, 1.9, 0.0, -1.0],
        [-0.5, 0.0, 0.2, 2.0, -0.2],
        [-0.8, -0.3, 0.0, 0.2, 1.7],
    ],
    dtype=np.float64,
)
# Next-token 训练目标是“当前位置预测下一个 token”，所以 target 是原序列向左错一位。
lm_targets = sequence[1:]
# 计算每个位置的 token loss 和平均 token loss。
lm_per_token_loss, lm_mean_loss, lm_probabilities = cross_entropy_batch(lm_logits, lm_targets)
# 打印原始 token 序列。
print("sequence token ids =", sequence)
# 打印可读 token 序列。
print("sequence tokens =", [vocab[i] for i in sequence])
# 打印模型 logits Shape。
print("lm_logits.shape =", lm_logits.shape)
# 打印 target Shape。
print("lm_targets.shape =", lm_targets.shape)
# 打印每个位置的 target token。
print("next-token targets =", [vocab[i] for i in lm_targets])
# 遍历每个预测位置，展示当前位置输入、下一个真实 token、真实 token 概率和 loss。
for position in range(lm_logits.shape[0]):
    # 当前输入 token 是 sequence[position]。
    current_token = vocab[sequence[position]]
    # 正确的下一个 token 是 lm_targets[position]。
    target_token = vocab[lm_targets[position]]
    # 取出正确下一个 token 的概率。
    p_target = lm_probabilities[position, lm_targets[position]]
    # 取出当前位置的 loss。
    token_loss = lm_per_token_loss[position]
    # 打印这一位置的训练信息。
    print(
        f"position={position}, input={current_token}, target={target_token}, "
        f"p_target={p_target:.6f}, token_loss={token_loss:.6f}"
    )
# 打印整个序列上平均 token loss。
print("mean next-token loss =", lm_mean_loss)

# 实验9：验证 LLM 常见 Shape (B,T,V) 可以展平为 (B*T,V) 后统一计算 Cross Entropy。
section("实验9：(B,T,V) logits 展平后计算 token Cross Entropy")
# 构造 Batch Size=2 的 toy logits，第二个样本在第一个样本基础上加一个小偏移。
lm_logits_btv = np.stack([lm_logits, lm_logits + np.array([0.1, -0.1, 0.0, 0.05, -0.05])], axis=0)
# 构造两个样本对应的 targets，Shape 是 (B=2,T=4)。
lm_targets_bt = np.stack([lm_targets, lm_targets], axis=0)
# 保存 B、T、V 三个维度。
B, T, V = lm_logits_btv.shape
# 把 logits 从 (B,T,V) 展平成 (B*T,V)。
flat_logits = lm_logits_btv.reshape(B * T, V)
# 把 targets 从 (B,T) 展平成 (B*T,)。
flat_targets = lm_targets_bt.reshape(B * T)
# 对展平后的 token 统一计算 Cross Entropy。
flat_losses, flat_mean_loss, _ = cross_entropy_batch(flat_logits, flat_targets)
# 打印原始 logits Shape。
print("lm_logits_btv.shape =", lm_logits_btv.shape)
# 打印原始 targets Shape。
print("lm_targets_bt.shape =", lm_targets_bt.shape)
# 打印展平后 logits Shape。
print("flat_logits.shape =", flat_logits.shape)
# 打印展平后 targets Shape。
print("flat_targets.shape =", flat_targets.shape)
# 打印8个 token 位置各自的 loss。
print("flat token losses =", flat_losses)
# 打印所有有效 token 的平均 loss。
print("flat mean token loss =", flat_mean_loss)

# 绘图1：正确类别 logit 越高，正确类别概率越高，Cross Entropy 越低。
fig, ax1 = plt.subplots(figsize=(8, 5))
# 用左轴画 Cross Entropy。
ax1.plot(correct_logits, losses, marker="o", label="Cross Entropy")
# 设置横轴标题。
ax1.set_xlabel("Correct-class logit")
# 设置左纵轴标题。
ax1.set_ylabel("Cross Entropy")
# 开启网格，便于观察趋势。
ax1.grid(True, alpha=0.3)
# 创建共享横轴的右纵轴。
ax2 = ax1.twinx()
# 用右轴画正确类别概率。
ax2.plot(correct_logits, correct_probabilities, marker="s", linestyle="--", label="p(correct)")
# 设置右纵轴标题。
ax2.set_ylabel("Probability of correct class")
# 设置图标题。
ax1.set_title("Correct logit vs probability and Cross Entropy")
# 调整布局，避免标签被裁切。
fig.tight_layout()
# 保存图片。
fig.savefig("figures/lesson18_correct_logit_vs_loss.png", dpi=180)
# 关闭图，释放内存。
plt.close(fig)

# 绘图2：正确类别概率与 -log(p) 的关系。
# 从0.001到1取一组概率，避免 log(0)。
p_values = np.linspace(0.001, 1.0, 500)
# 计算每个概率对应的负对数损失。
nll_values = -np.log(p_values)
# 创建图。
fig, ax = plt.subplots(figsize=(8, 5))
# 画出概率与 loss 的曲线。
ax.plot(p_values, nll_values)
# 设置横轴标题。
ax.set_xlabel("Probability assigned to the correct class")
# 设置纵轴标题。
ax.set_ylabel("Cross Entropy = -log(p_correct)")
# 设置图标题。
ax.set_title("Lower correct probability receives larger penalty")
# 开启网格。
ax.grid(True, alpha=0.3)
# 调整布局。
fig.tight_layout()
# 保存图片。
fig.savefig("figures/lesson18_probability_vs_cross_entropy.png", dpi=180)
# 关闭图。
plt.close(fig)

# 绘图3：微型 LLM 四个位置的 token loss。
# 创建图。
fig, ax = plt.subplots(figsize=(8, 4.8))
# 构造位置索引。
positions = np.arange(len(lm_per_token_loss))
# 绘制柱状图。
ax.bar(positions, lm_per_token_loss)
# 设置横轴刻度。
ax.set_xticks(positions)
# 设置横轴标签为 A->B、B->C 等。
ax.set_xticklabels([f"{vocab[sequence[i]]}->{vocab[lm_targets[i]]}" for i in positions])
# 设置横轴标题。
ax.set_xlabel("Next-token prediction position")
# 设置纵轴标题。
ax.set_ylabel("Token Cross Entropy")
# 设置图标题。
ax.set_title("Toy LLM: per-token next-token loss")
# 开启横向网格。
ax.grid(True, axis="y", alpha=0.3)
# 调整布局。
fig.tight_layout()
# 保存图片。
fig.savefig("figures/lesson18_next_token_losses.png", dpi=180)
# 关闭图。
plt.close(fig)

# 打印结束信息。
section("实验完成")
# 提示生成的三张图片路径。
print("已生成 figures/lesson18_correct_logit_vs_loss.png")
# 提示第二张图片。
print("已生成 figures/lesson18_probability_vs_cross_entropy.png")
# 提示第三张图片。
print("已生成 figures/lesson18_next_token_losses.png")
