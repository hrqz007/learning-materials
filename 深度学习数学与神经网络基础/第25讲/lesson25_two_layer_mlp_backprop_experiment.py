# 第25讲实验：两层 MLP 的 Forward / Backpropagation / Gradient Check / PyTorch 对照
import numpy as np  # 导入 NumPy，用来完成矩阵运算和手工反向传播
from pathlib import Path  # 导入 Path，用来稳定定位输出文件夹

OUT_DIR = Path(__file__).resolve().parent  # 取得当前脚本所在目录，所有实验输出都放在这里
FIG_DIR = OUT_DIR / 'figures'  # 指定实验图片保存目录
FIG_DIR.mkdir(exist_ok=True)  # 如果 figures 目录不存在就创建它
np.set_printoptions(precision=6, suppress=True)  # 设置 NumPy 打印精度，避免科学计数法干扰初学者阅读


def relu(x):  # 定义 ReLU 前向函数，输入和输出 Shape 完全相同
    return np.maximum(0.0, x)  # 小于 0 的元素变成 0，大于 0 的元素保持不变


def relu_grad(x):  # 定义 ReLU 的局部导数函数，用于手工 Backward
    return (x > 0).astype(float)  # x>0 的位置导数为 1，否则为 0


def forward(x, W1, b1, W2, b2, target):  # 定义两层 MLP 的完整 Forward
    z1 = x @ W1 + b1  # 第一层 Linear：把输入从 Din=3 映射到 H=2
    a1 = relu(z1)  # 第一层激活：对 z1 逐元素执行 ReLU
    y_hat = a1 @ W2 + b2  # 第二层 Linear：把 H=2 映射到 Dout=1
    error = y_hat - target  # 计算预测与真实目标之间的误差
    loss = np.sum(error ** 2)  # 使用平方误差，并把所有输出元素求和得到标量 Loss
    cache = {'x': x, 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'z1': z1, 'a1': a1, 'y_hat': y_hat, 'target': target, 'error': error}  # 缓存 Backward 所需的中间量
    return float(loss), cache  # 返回标量 Loss 和中间变量缓存


def backward(cache):  # 定义两层 MLP 的手工矩阵 Backpropagation
    x = cache['x']  # 取出第一层输入 x，后面计算 dW1 要使用它
    W1 = cache['W1']  # 取出第一层 Weight，后面计算 dx 要使用它
    W2 = cache['W2']  # 取出第二层 Weight，后面计算 da1 要使用它
    z1 = cache['z1']  # 取出 ReLU 之前的 z1，用来计算 ReLU mask
    a1 = cache['a1']  # 取出第二层输入 a1，用来计算 dW2
    error = cache['error']  # 取出 y_hat-target，用来计算 Loss 对输出的梯度
    dy_hat = 2.0 * error  # 对 L=(y_hat-target)^2 求导，得到 dL/dy_hat
    dW2 = a1.T @ dy_hat  # 第二层参数梯度：输入转置乘以上游梯度，Shape 为 (2,1)
    db2 = np.sum(dy_hat, axis=0)  # 第二层 Bias 梯度：沿 batch 维把输出梯度相加
    da1 = dy_hat @ W2.T  # 把梯度通过第二层 Weight 的转置传回隐藏层，Shape 为 (1,2)
    dz1 = da1 * relu_grad(z1)  # 穿过 ReLU gate，负区间的梯度被乘成 0
    dW1 = x.T @ dz1  # 第一层参数梯度：第一层输入转置乘以上游梯度，Shape 为 (3,2)
    db1 = np.sum(dz1, axis=0)  # 第一层 Bias 梯度：沿 batch 维把隐藏层梯度相加
    dx = dz1 @ W1.T  # 把梯度继续传给更前面的输入，Shape 回到 (1,3)
    return {'dy_hat': dy_hat, 'dW2': dW2, 'db2': db2, 'da1': da1, 'dz1': dz1, 'dW1': dW1, 'db1': db1, 'dx': dx}  # 返回所有关键梯度


def central_difference(param, idx, loss_fn, eps=1e-5):  # 定义中心差分，用数值方法检查某个参数元素的梯度
    old = param[idx]  # 暂存这个参数元素原来的值，实验结束后要恢复
    param[idx] = old + eps  # 把当前参数元素轻微增加 eps
    loss_plus = loss_fn()  # 重新 Forward，记录参数增加后的 Loss
    param[idx] = old - eps  # 把当前参数元素轻微减少 eps
    loss_minus = loss_fn()  # 重新 Forward，记录参数减少后的 Loss
    param[idx] = old  # 把参数恢复成原值，避免污染后续实验
    return (loss_plus - loss_minus) / (2.0 * eps)  # 用中心差分公式近似该位置的偏导数


x = np.array([[1.0, 2.0, -1.0]])  # 构造 1 个样本、3 个输入特征，Shape=(1,3)
W1 = np.array([[0.5, -1.0], [1.0, 0.5], [-0.5, 2.0]])  # 第一层 Weight，Shape=(3,2)
b1 = np.array([0.0, 0.0])  # 第一层 Bias，2 个隐藏神经元各有 1 个 Bias
W2 = np.array([[2.0], [-1.0]])  # 第二层 Weight，Shape=(2,1)
b2 = np.array([0.5])  # 第二层 Bias，Shape=(1,)
target = np.array([[5.0]])  # 真实目标值，Shape=(1,1)

loss, cache = forward(x, W1, b1, W2, b2, target)  # 执行一次完整 Forward，得到 Loss 和中间缓存
grads = backward(cache)  # 使用缓存执行一次完整手工 Backward

lines = []  # 创建列表，用来同时保存终端输出到 TXT 文件

def log(text=''):  # 定义统一输出函数，让屏幕输出和 TXT 内容完全一致
    print(text)  # 把信息打印到终端
    lines.append(str(text))  # 同时把相同信息保存到列表中

log('=== 第25讲：两层 MLP Backpropagation 实验 ===')  # 输出实验标题
log()  # 输出空行，提升可读性
log('1) 固定网络与 Forward')  # 输出实验1标题
for name, arr in [('x', x), ('W1', W1), ('b1', b1), ('W2', W2), ('b2', b2), ('target', target)]:  # 依次遍历输入、参数和目标
    log(f'{name}.shape = {arr.shape}, value =\n{arr}')  # 打印每个对象的 Shape 与数值
log(f'z1 = x @ W1 + b1 = {cache["z1"]}')  # 打印第一层 Linear 输出
log(f'a1 = ReLU(z1) = {cache["a1"]}')  # 打印 ReLU 后的隐藏表示
log(f'y_hat = a1 @ W2 + b2 = {cache["y_hat"]}')  # 打印最终预测值
log(f'error = {cache["error"]}')  # 打印预测误差
log(f'loss = {loss:.6f}')  # 打印标量 Loss

log('\n2) 手工矩阵 Backward')  # 输出实验2标题
for key in ['dy_hat', 'dW2', 'db2', 'da1', 'dz1', 'dW1', 'db1', 'dx']:  # 按反向传播顺序遍历关键梯度
    value = grads[key]  # 取出当前梯度 Tensor
    log(f'{key}.shape = {value.shape}, value =\n{value}')  # 打印当前梯度的 Shape 与数值

log('\n3) 关键 Shape 关系')  # 输出实验3标题
for name, grad in [('W1', grads['dW1']), ('b1', grads['db1']), ('W2', grads['dW2']), ('b2', grads['db2']), ('x', grads['dx'])]:  # 遍历参数/输入与对应梯度
    param = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'x': x}[name]  # 根据名字取得原始 Tensor
    log(f'{name}.shape == dL/d{name}.shape ? {param.shape} == {grad.shape} -> {param.shape == grad.shape}')  # 检查 Gradient Shape 是否匹配

log('\n4) Gradient Check')  # 输出实验4标题
def loss_fn():  # 定义无参数函数，供 Numerical Gradient Check 反复重新计算 Loss
    return forward(x, W1, b1, W2, b2, target)[0]  # 返回当前参数状态下的标量 Loss

checks = []  # 创建列表，保存每一个参数元素的解析梯度与数值梯度
for name, param, analytic in [('W1', W1, grads['dW1']), ('b1', b1, grads['db1']), ('W2', W2, grads['dW2']), ('b2', b2, grads['db2']), ('x', x, grads['dx'])]:  # 遍历所有需要检查的 Tensor
    iterator = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])  # 创建逐元素迭代器，并记录每个元素的多维索引
    while not iterator.finished:  # 只要当前 Tensor 还有元素没有检查就继续
        idx = iterator.multi_index  # 取得当前元素的多维索引
        numeric = central_difference(param, idx, loss_fn)  # 用中心差分得到当前元素的数值梯度
        analytic_value = analytic[idx]  # 读取手工 Backprop 得到的解析梯度
        abs_error = abs(numeric - analytic_value)  # 计算解析梯度与数值梯度之间的绝对误差
        checks.append((name, idx, analytic_value, numeric, abs_error))  # 保存当前检查结果，后面画图使用
        log(f'{name}{idx}: analytic={analytic_value:.9f}, numeric={numeric:.9f}, abs_err={abs_error:.3e}')  # 打印当前元素的对照结果
        iterator.iternext()  # 移动到当前 Tensor 的下一个元素
log(f'max gradient-check error = {max(item[-1] for item in checks):.3e}')  # 输出所有检查中的最大绝对误差

log('\n5) ReLU gate 对前层梯度的影响')  # 输出实验5标题
log(f'z1 = {cache["z1"]}; ReLU mask = {relu_grad(cache["z1"])}')  # 打印 Forward 时的 z1 与 ReLU mask
log(f'da1 = {grads["da1"]}')  # 打印进入 ReLU 后方的上游梯度
log(f'dz1 = da1 * mask = {grads["dz1"]}')  # 打印穿过 ReLU 后的梯度
log('第二个 hidden unit 的 z1<0，因此它对应的 W1 第二列梯度全部为0。')  # 解释第二个隐藏单元为什么阻断前层梯度

log('\n6) 做一次小步参数更新')  # 输出实验6标题
learning_rate = 0.01  # 设置很小的学习率，减少跨越局部有效区域的风险
W1_new = W1 - learning_rate * grads['dW1']  # 沿负梯度方向更新第一层 Weight
b1_new = b1 - learning_rate * grads['db1']  # 沿负梯度方向更新第一层 Bias
W2_new = W2 - learning_rate * grads['dW2']  # 沿负梯度方向更新第二层 Weight
b2_new = b2 - learning_rate * grads['db2']  # 沿负梯度方向更新第二层 Bias
loss_new, _ = forward(x, W1_new, b1_new, W2_new, b2_new, target)  # 使用更新后的参数重新 Forward，检查 Loss 是否下降
log(f'learning rate = {learning_rate}')  # 打印本次参数更新使用的学习率
log(f'loss before = {loss:.6f}')  # 打印参数更新前的 Loss
log(f'loss after  = {loss_new:.6f}')  # 打印参数更新后的 Loss

log('\n7) Batch 版本预览：mean MSE')  # 输出实验7标题
X_batch = np.array([[1.0, 2.0, -1.0], [0.0, 1.0, 1.0]])  # 构造 2 个样本组成的 Batch，Shape=(2,3)
Y_batch = np.array([[5.0], [1.0]])  # 构造两个样本对应的目标，Shape=(2,1)

def batch_forward_backward(X, W1, b1, W2, b2, Y):  # 定义 Batch 版本的 Forward 与 Backward
    Z1 = X @ W1 + b1  # 对整个 Batch 一次完成第一层 Linear
    A1 = relu(Z1)  # 对整个 Batch 的隐藏表示逐元素执行 ReLU
    Y_hat = A1 @ W2 + b2  # 对整个 Batch 一次完成第二层 Linear
    error = Y_hat - Y  # 计算 Batch 中每个样本的预测误差
    batch_loss = np.mean(error ** 2)  # 对两个样本的平方误差取平均，得到 mean MSE
    batch_size = X.shape[0]  # 读取 Batch Size，mean MSE 的梯度需要除以它
    dY_hat = (2.0 / batch_size) * error  # 对 mean MSE 求导，得到每个样本的输出梯度
    dW2 = A1.T @ dY_hat  # 把所有样本对第二层 Weight 的梯度贡献通过矩阵乘法聚合
    db2 = np.sum(dY_hat, axis=0)  # 沿 batch 维聚合第二层 Bias 梯度
    dA1 = dY_hat @ W2.T  # 把 Batch 的梯度通过第二层 Weight 转置传回隐藏层
    dZ1 = dA1 * relu_grad(Z1)  # 对 Batch 中每个 hidden value 应用自己的 ReLU mask
    dW1 = X.T @ dZ1  # 把所有样本对第一层 Weight 的梯度贡献聚合成一个矩阵
    db1 = np.sum(dZ1, axis=0)  # 沿 batch 维聚合第一层 Bias 梯度
    dX = dZ1 @ W1.T  # 把梯度继续传回 Batch 中每个样本的输入
    return batch_loss, {'Z1': Z1, 'A1': A1, 'Y_hat': Y_hat, 'error': error, 'dY_hat': dY_hat, 'dW2': dW2, 'db2': db2, 'dA1': dA1, 'dZ1': dZ1, 'dW1': dW1, 'db1': db1, 'dX': dX}  # 返回 Batch Loss 与关键中间量

batch_loss, batch_grads = batch_forward_backward(X_batch, W1, b1, W2, b2, Y_batch)  # 执行 Batch 版本的 Forward 与 Backward
log(f'X_batch.shape={X_batch.shape}, Y_batch.shape={Y_batch.shape}, batch_loss={batch_loss:.6f}')  # 打印 Batch Shape 与 Loss
log(f'Z1=\n{batch_grads["Z1"]}')  # 打印两个样本在第一层的 pre-activation
log(f'ReLU mask=\n{relu_grad(batch_grads["Z1"])}')  # 打印两个样本各自不同的 ReLU gate
log(f'dW1.shape={batch_grads["dW1"].shape}, dW1=\n{batch_grads["dW1"]}')  # 打印 Batch 聚合后的第一层 Weight 梯度
log(f'dW2.shape={batch_grads["dW2"].shape}, dW2=\n{batch_grads["dW2"]}')  # 打印 Batch 聚合后的第二层 Weight 梯度

log('\n8) PyTorch Autograd 对照')  # 输出实验8标题
try:  # 尝试导入并运行 PyTorch；如果当前环境没有安装则跳过而不是让整个实验失败
    import torch  # 导入 PyTorch，用 Autograd 独立验证手工 NumPy 梯度
    x_t = torch.tensor(x, dtype=torch.float64, requires_grad=True)  # 把输入转成需要梯度的 PyTorch Tensor
    W1_t = torch.tensor(W1, dtype=torch.float64, requires_grad=True)  # 把第一层 Weight 转成可求导 Tensor
    b1_t = torch.tensor(b1, dtype=torch.float64, requires_grad=True)  # 把第一层 Bias 转成可求导 Tensor
    W2_t = torch.tensor(W2, dtype=torch.float64, requires_grad=True)  # 把第二层 Weight 转成可求导 Tensor
    b2_t = torch.tensor(b2, dtype=torch.float64, requires_grad=True)  # 把第二层 Bias 转成可求导 Tensor
    target_t = torch.tensor(target, dtype=torch.float64)  # 把目标转成普通 Tensor，目标本身不需要训练
    z1_t = x_t @ W1_t + b1_t  # 用 PyTorch 执行第一层 Linear Forward
    a1_t = torch.relu(z1_t)  # 用 PyTorch 执行 ReLU Forward
    y_hat_t = a1_t @ W2_t + b2_t  # 用 PyTorch 执行第二层 Linear Forward
    loss_t = torch.sum((y_hat_t - target_t) ** 2)  # 使用与 NumPy 完全相同的 sum squared error
    loss_t.backward()  # 让 Autograd 沿计算图自动执行完整 Backpropagation
    log(f'torch loss = {loss_t.item():.6f}')  # 打印 PyTorch Forward 得到的 Loss
    log(f'W1.grad=\n{W1_t.grad.detach().numpy()}')  # 打印 PyTorch 计算的第一层 Weight 梯度
    log(f'b1.grad={b1_t.grad.detach().numpy()}')  # 打印 PyTorch 计算的第一层 Bias 梯度
    log(f'W2.grad=\n{W2_t.grad.detach().numpy()}')  # 打印 PyTorch 计算的第二层 Weight 梯度
    log(f'b2.grad={b2_t.grad.detach().numpy()}')  # 打印 PyTorch 计算的第二层 Bias 梯度
    log(f'x.grad=\n{x_t.grad.detach().numpy()}')  # 打印 PyTorch 传回输入的梯度
    diffs = [np.max(np.abs(W1_t.grad.detach().numpy() - grads['dW1'])), np.max(np.abs(b1_t.grad.detach().numpy() - grads['db1'])), np.max(np.abs(W2_t.grad.detach().numpy() - grads['dW2'])), np.max(np.abs(b2_t.grad.detach().numpy() - grads['db2'])), np.max(np.abs(x_t.grad.detach().numpy() - grads['dx']))]  # 计算 NumPy 手工结果与 PyTorch 的最大差异
    log(f'max NumPy-vs-PyTorch gradient diff = {max(diffs):.3e}')  # 打印所有梯度中的最大实现差异
except Exception as error:  # 捕获 PyTorch 不可用或运行失败的情况
    log(f'PyTorch unavailable or failed: {error}')  # 打印原因，但不影响前面的 NumPy 实验结果

import matplotlib.pyplot as plt  # 导入 Matplotlib，用来把关键实验结果画成图片
from matplotlib import font_manager  # 导入字体管理器，解决中文图题乱码问题
font_path = '/usr/share/fonts/truetype/arphic-gbsn00lp/gbsn00lp.ttf'  # 指定系统中的中文字体文件
font_prop = font_manager.FontProperties(fname=font_path)  # 从字体文件创建 Matplotlib 字体对象
plt.rcParams['font.family'] = font_prop.get_name()  # 让本实验的所有图默认使用中文字体
plt.rcParams['axes.unicode_minus'] = False  # 让坐标轴负号正常显示而不是方块

fig, ax = plt.subplots(figsize=(12, 5.5))  # 创建 Forward/Backward 总览图画布
ax.axis('off')  # 关闭坐标轴，因为这张图是流程图而不是数据坐标图
positions = [0.07, 0.29, 0.50, 0.72, 0.91]  # 给五个 Forward/Backward 节点设置横向位置
forward_labels = ['x\n(1,3)', 'z1=xW1+b1\n[3,-2]', 'a1=ReLU(z1)\n[3,0]', 'y_hat\n6.5', 'Loss\n2.25']  # 定义 Forward 路径上的文字
for i, (pos, label) in enumerate(zip(positions, forward_labels)):  # 逐个绘制 Forward 节点
    ax.text(pos, 0.70, label, ha='center', va='center', bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='black'))  # 绘制当前 Forward 节点文本框
    if i < len(positions) - 1:  # 如果当前节点不是最后一个，就画箭头指向下一个节点
        ax.annotate('', xy=(positions[i + 1] - 0.07, 0.70), xytext=(pos + 0.07, 0.70), arrowprops=dict(arrowstyle='->'))  # 绘制 Forward 方向箭头
backward_labels = [('dX\n[3,6,-3]', 0.07), ('dZ1\n[6,0]', 0.29), ('dA1\n[6,-3]', 0.50), ('dY_hat\n3', 0.72), ('1', 0.91)]  # 定义 Backward 路径上的梯度文字
for label, pos in backward_labels:  # 逐个绘制 Backward 节点
    ax.text(pos, 0.25, label, ha='center', va='center', bbox=dict(boxstyle='round,pad=0.45', fc='white', ec='black'))  # 绘制当前 Backward 节点文本框
for i in range(len(positions) - 1, 0, -1):  # 从右向左遍历节点位置，准备绘制反向箭头
    ax.annotate('', xy=(positions[i - 1] + 0.07, 0.25), xytext=(positions[i] - 0.07, 0.25), arrowprops=dict(arrowstyle='->'))  # 绘制 Backward 方向箭头
ax.text(0.5, 0.92, '两层 MLP：上方 Forward，下方 Backward', ha='center', va='center', fontsize=15)  # 添加总览图标题
fig.tight_layout()  # 自动调整图中元素，减少文字被裁切的风险
fig.savefig(FIG_DIR / 'lesson25_mlp_forward_backward.png', dpi=180, bbox_inches='tight')  # 保存 Forward/Backward 总览图
plt.close(fig)  # 关闭当前图，释放绘图资源

names = []  # 创建列表，保存 Gradient Check 横轴标签
analytic_values = []  # 创建列表，保存解析梯度
numeric_values = []  # 创建列表，保存数值梯度
for name, idx, analytic_value, numeric, abs_error in checks:  # 遍历前面保存的所有逐元素 Gradient Check 结果
    names.append(f'{name}{idx}')  # 把参数名和索引组合成横轴标签
    analytic_values.append(analytic_value)  # 保存解析梯度数值
    numeric_values.append(numeric)  # 保存数值差分梯度数值
fig, ax = plt.subplots(figsize=(12, 5))  # 创建 Gradient Check 对比柱状图画布
positions_bar = np.arange(len(names))  # 为每一个参数元素生成一个横轴位置
bar_width = 0.38  # 设置两组柱子的宽度
ax.bar(positions_bar - bar_width / 2, analytic_values, bar_width, label='analytic')  # 绘制手工解析梯度柱子
ax.bar(positions_bar + bar_width / 2, numeric_values, bar_width, label='numeric')  # 绘制数值差分梯度柱子
ax.set_xticks(positions_bar)  # 设置横轴刻度位置
ax.set_xticklabels(names, rotation=55, ha='right')  # 设置横轴参数标签并旋转，避免重叠
ax.set_ylabel('gradient')  # 设置纵轴含义
ax.set_title('手工矩阵 Backprop 与 Numerical Gradient Check')  # 设置图标题
ax.legend()  # 显示 analytic 与 numeric 图例
ax.axhline(0, linewidth=0.8)  # 绘制 y=0 基准线，方便观察正负梯度
fig.tight_layout()  # 自动调整版面，避免标签被裁切
fig.savefig(FIG_DIR / 'lesson25_gradient_check.png', dpi=180, bbox_inches='tight')  # 保存 Gradient Check 图
plt.close(fig)  # 关闭当前图，释放绘图资源

fig, ax = plt.subplots(figsize=(7, 5))  # 创建 ReLU gate 对第一层梯度影响图
column_norms = np.linalg.norm(grads['dW1'], axis=0)  # 计算 dW1 每一列的 L2 范数，对应两个 hidden unit 的 incoming gradient 强度
ax.bar(['hidden unit 1\n(z1=3)', 'hidden unit 2\n(z1=-2)'], column_norms)  # 绘制两个 hidden unit 对应的第一层梯度列范数
ax.set_ylabel('||dW1 column||')  # 设置纵轴为第一层 Weight 梯度列范数
ax.set_title('ReLU gate 对前一层 Weight 梯度的影响')  # 设置中文图标题
fig.tight_layout()  # 自动调整版面，防止文字裁切
fig.savefig(FIG_DIR / 'lesson25_relu_gate_columns.png', dpi=180, bbox_inches='tight')  # 保存 ReLU gate 实验图
plt.close(fig)  # 关闭当前图，释放绘图资源

fig, ax = plt.subplots(figsize=(6, 4.5))  # 创建单步参数更新前后 Loss 对比图
ax.bar(['before', 'after'], [loss, loss_new])  # 绘制更新前后两个 Loss 柱子
ax.set_ylabel('Loss')  # 设置纵轴含义为 Loss
ax.set_title('一次小步负梯度更新')  # 设置图标题
for i, value in enumerate([loss, loss_new]):  # 遍历两个 Loss 值，为柱子标注具体数字
    ax.text(i, value + 0.04, f'{value:.4f}', ha='center')  # 在每个柱子上方写出 Loss 数值
fig.tight_layout()  # 自动调整版面，防止标题或标签被裁切
fig.savefig(FIG_DIR / 'lesson25_one_step_update.png', dpi=180, bbox_inches='tight')  # 保存单步更新结果图
plt.close(fig)  # 关闭当前图，释放绘图资源

(OUT_DIR / '第25讲_实验实际输出.txt').write_text('\n'.join(lines), encoding='utf-8')  # 把所有实验终端输出保存成 UTF-8 TXT，方便复核与复现
