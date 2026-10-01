import math  # 导入 math，使用自然对数等基础数学函数
import numpy as np  # 导入 NumPy，完成数组和指数/对数计算

try:  # 尝试导入 PyTorch，用于和 NumPy 做结果核对
    import torch  # 导入 PyTorch
    TORCH_AVAILABLE = True  # 记录 PyTorch 可以使用
except ImportError:  # 如果当前环境没有安装 PyTorch
    TORCH_AVAILABLE = False  # 记录 PyTorch 不可使用


def check_probability_distribution(p, name):  # 定义函数：检查一个向量是不是合法概率分布
    p = np.asarray(p, dtype=np.float64)  # 把输入转换为浮点 NumPy 数组
    in_range = np.all((p >= 0.0) & (p <= 1.0))  # 检查每个概率是否都在 0 到 1 之间
    sums_to_one = np.isclose(np.sum(p), 1.0)  # 检查所有概率之和是否约等于 1
    print(f"{name}: {p}")  # 打印概率向量
    print(f"  每项都在[0,1]：{in_range}")  # 打印范围检查结果
    print(f"  概率和：{np.sum(p):.6f}")  # 打印概率总和
    print(f"  是否为合法离散概率分布：{in_range and sums_to_one}")  # 打印综合判断


print("=" * 70)  # 打印分隔线
print("实验1：概率分布为什么必须满足 0<=p<=1 且总和为1")  # 打印实验标题
print("=" * 70)  # 打印分隔线
check_probability_distribution([0.2, 0.3, 0.5], "合法分布")  # 检查一个合法概率分布
check_probability_distribution([0.2, 0.3, 0.4], "总和不足1")  # 检查总和不是 1 的向量
check_probability_distribution([-0.1, 0.4, 0.7], "包含负数")  # 检查包含负概率的向量

print("\n" + "=" * 70)  # 打印空行和分隔线
print("实验2：指数函数 e^x 如何改变数字差异")  # 打印实验标题
print("=" * 70)  # 打印分隔线
x_values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])  # 定义一组输入值
exp_values = np.exp(x_values)  # 对每个输入计算 e^x
for x, y in zip(x_values, exp_values):  # 逐个遍历输入和指数结果
    print(f"x={x:>4.1f} -> e^x={y:.6f}")  # 打印每个输入对应的指数值
ratio = np.exp(2.0) / np.exp(1.0)  # 计算 e^2 与 e^1 的比值
print(f"e^2 / e^1 = {ratio:.6f}")  # 打印指数比值
print(f"e^(2-1)   = {np.exp(1.0):.6f}")  # 验证 e^a/e^b=e^(a-b)

print("\n" + "=" * 70)  # 打印空行和分隔线
print("实验3：log 是 exp 的逆运算")  # 打印实验标题
print("=" * 70)  # 打印分隔线
positive_values = np.array([0.5, 1.0, 2.0, 5.0])  # 定义只能取正数的 log 输入
log_values = np.log(positive_values)  # 计算自然对数 ln(x)
recovered_values = np.exp(log_values)  # 再做 exp(log(x)) 恢复原值
for x, l, r in zip(positive_values, log_values, recovered_values):  # 遍历每个结果
    print(f"x={x:.1f}, ln(x)={l:.6f}, exp(ln(x))={r:.6f}")  # 打印逆运算验证

print("\n" + "=" * 70)  # 打印空行和分隔线
print("实验4：-log(p) 如何把低概率变成大惩罚")  # 打印实验标题
print("=" * 70)  # 打印分隔线
probabilities = np.array([0.90, 0.50, 0.10, 0.01])  # 定义模型给正确答案的几种概率
negative_log = -np.log(probabilities)  # 计算负对数惩罚
for p, loss in zip(probabilities, negative_log):  # 遍历概率和对应惩罚
    print(f"p={p:.2f} -> -ln(p)={loss:.6f}")  # 打印概率越小惩罚越大的现象

print("\n" + "=" * 70)  # 打印空行和分隔线
print("实验5：序列概率连乘为什么很快变小，log 为什么有用")  # 打印实验标题
print("=" * 70)  # 打印分隔线
conditional_probs = np.array([0.8, 0.7, 0.9])  # 定义三个连续 token 的条件概率
sequence_prob = np.prod(conditional_probs)  # 把条件概率相乘得到整个序列概率
sum_log_prob = np.sum(np.log(conditional_probs))  # 把每个概率取 log 后求和
negative_log_likelihood = -sum_log_prob  # 对 log 概率和取负号得到 NLL
print(f"三个条件概率：{conditional_probs}")  # 打印条件概率
print(f"序列概率乘积：{sequence_prob:.6f}")  # 打印序列概率
print(f"log(序列概率)：{np.log(sequence_prob):.6f}")  # 直接计算序列概率的 log
print(f"各 log 概率之和：{sum_log_prob:.6f}")  # 打印 log 之和
print(f"负对数似然 NLL：{negative_log_likelihood:.6f}")  # 打印 NLL

same_token_p = 0.8  # 假设每一步正确 token 的条件概率都为 0.8
for n in [1, 5, 10, 20, 30]:  # 选择不同序列长度
    product_prob = same_token_p ** n  # 计算 n 个 0.8 连乘
    cumulative_nll = -n * math.log(same_token_p)  # 用 log 把连乘改写成加法
    print(f"长度={n:>2d}: 概率乘积={product_prob:.10f}, 累积NLL={cumulative_nll:.6f}")  # 打印结果

print("\n" + "=" * 70)  # 打印空行和分隔线
print("实验6：为什么 raw score 不能直接当概率；指数能先把它变成正数")  # 打印实验标题
print("=" * 70)  # 打印分隔线
scores = np.array([-2.0, 1.0, 0.1])  # 定义包含负数的 raw scores/logits
naive = scores / np.sum(scores)  # 直接除以总和，观察会得到非法概率
exp_scores = np.exp(scores)  # 对 raw score 取指数，使每一项都变成正数
exp_normalized = exp_scores / np.sum(exp_scores)  # 再除以总和，得到合法概率分布
print(f"raw scores：{scores}")  # 打印原始分数
print(f"直接除以总和：{naive}")  # 打印错误的“直接归一化”结果
print(f"exp(scores)：{exp_scores}")  # 打印指数结果
print(f"exp 后归一化：{exp_normalized}")  # 打印指数归一化后的概率
print(f"概率和：{np.sum(exp_normalized):.6f}")  # 验证概率和为 1

print("\n" + "=" * 70)  # 打印空行和分隔线
print("实验7：指数的数值稳定性——为什么会先减最大值")  # 打印实验标题
print("=" * 70)  # 打印分隔线
large_scores = np.array([1000.0, 1001.0, 999.0])  # 定义很大的 raw scores
with np.errstate(over='ignore', invalid='ignore'):  # 暂时关闭指数溢出的警告输出
    naive_exp = np.exp(large_scores)  # 直接计算指数，可能得到 inf
    naive_probs = naive_exp / np.sum(naive_exp)  # 直接归一化，可能出现 nan
shifted_scores = large_scores - np.max(large_scores)  # 给所有 score 同时减去最大值
stable_exp = np.exp(shifted_scores)  # 对平移后的 score 计算指数
stable_probs = stable_exp / np.sum(stable_exp)  # 对稳定指数结果归一化
print(f"large scores：{large_scores}")  # 打印大分数
print(f"直接 exp：{naive_exp}")  # 打印直接指数结果
print(f"直接归一化：{naive_probs}")  # 打印可能出现 nan 的结果
print(f"减最大值后 scores：{shifted_scores}")  # 打印稳定化后的分数
print(f"稳定归一化概率：{stable_probs}")  # 打印稳定概率
print(f"稳定概率和：{np.sum(stable_probs):.6f}")  # 验证概率和为 1

print("\n" + "=" * 70)  # 打印空行和分隔线
print("实验8：NumPy 与 PyTorch 的 exp/log 是否一致")  # 打印实验标题
print("=" * 70)  # 打印分隔线
if TORCH_AVAILABLE:  # 只有安装 PyTorch 时才执行
    torch_x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float64)  # 创建 PyTorch Tensor
    torch_exp = torch.exp(torch_x)  # 用 PyTorch 计算指数
    numpy_exp = np.exp(torch_x.numpy())  # 用 NumPy 计算同样的指数
    exp_diff = np.max(np.abs(torch_exp.numpy() - numpy_exp))  # 计算最大绝对误差
    torch_p = torch.tensor([0.9, 0.5, 0.1, 0.01], dtype=torch.float64)  # 创建概率 Tensor
    torch_neglog = -torch.log(torch_p)  # 用 PyTorch 计算负对数
    numpy_neglog = -np.log(torch_p.numpy())  # 用 NumPy 计算负对数
    log_diff = np.max(np.abs(torch_neglog.numpy() - numpy_neglog))  # 计算最大绝对误差
    print(f"exp 最大绝对误差：{exp_diff:.12f}")  # 打印 exp 对照误差
    print(f"-log 最大绝对误差：{log_diff:.12f}")  # 打印 -log 对照误差
else:  # 如果没有安装 PyTorch
    print("当前环境未安装 PyTorch，跳过 PyTorch 对照实验。")  # 给出说明
