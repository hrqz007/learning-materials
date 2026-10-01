"""
第7讲：玩具 Attention 实验
目的：不使用深度学习框架，只观察 Attention 最核心的三步：
1) 计算相关性分数
2) softmax 变成权重
3) 按权重对 value 做加权求和

注意：这不是完整 Transformer，也没有真正的 Q/K/V 投影矩阵。
"""
import math


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def softmax(xs):
    # 减最大值是常见的数值稳定写法
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [x / s for x in exps]


def weighted_sum(weights, values):
    dim = len(values[0])
    out = [0.0] * dim
    for w, v in zip(weights, values):
        for j in range(dim):
            out[j] += w * v[j]
    return out


# 当前“查询”想找什么（教学用二维向量）
query = [1.0, 0.0]

# 三个位置的“标签/特征”
keys = [
    [2.0, 0.0],   # 与 query 很相似
    [0.5, 1.0],   # 中等相似
    [-0.2, 0.8],  # 较不相似
]

# 真正要读取的内容
values = [
    [10.0, 0.0],
    [0.0, 10.0],
    [5.0, 5.0],
]

scores = [dot(query, k) for k in keys]
weights = softmax(scores)
context = weighted_sum(weights, values)

print('query   =', query)
print('scores  =', [round(x, 4) for x in scores])
print('weights =', [round(x, 4) for x in weights])
print('sum(weights) =', round(sum(weights), 6))
print('context =', [round(x, 4) for x in context])

print('\n逐位置贡献：')
for i, (w, v) in enumerate(zip(weights, values), start=1):
    contribution = [w * x for x in v]
    print(f'位置{i}: weight={w:.4f}, value={v}, contribution={[round(x,4) for x in contribution]}')

print('\n观察：分数越高 -> softmax 权重越大 -> 该位置的 value 对 context 贡献越大。')
