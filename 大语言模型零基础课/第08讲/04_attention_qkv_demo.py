"""
第8讲：Attention / QKV 玩具实验
只用 Python 标准库，目的不是实现工业级 Transformer，
而是把 scores -> softmax -> weighted values -> causal mask 打印出来。
"""
import math


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def softmax(xs):
    m = max(xs)
    exps = [math.exp(x-m) for x in xs]
    s = sum(exps)
    return [e/s for e in exps]


def weighted_sum(weights, values):
    d = len(values[0])
    out = [0.0]*d
    for w, v in zip(weights, values):
        for j in range(d):
            out[j] += w*v[j]
    return out


print('=== 实验 A：一个 Query 看三个位置 ===')
Q = [1.0, 0.0]
K = [[2.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
V = [[10.0, 0.0], [0.0, 10.0], [5.0, 5.0]]

scores = [dot(Q, k) for k in K]
weights = softmax(scores)
out = weighted_sum(weights, V)

print('Q      =', Q)
print('scores =', [round(x, 4) for x in scores])
print('weights=', [round(x, 4) for x in weights], 'sum=', round(sum(weights), 6))
print('output =', [round(x, 4) for x in out])
print('\n逐位置贡献：')
for i, (w, v) in enumerate(zip(weights, V)):
    contrib = [w*x for x in v]
    print(f'pos{i}: weight={w:.4f}, V={v}, contribution={[round(x,4) for x in contrib]}')


print('\n=== 实验 B：Causal Mask ===')
# 4 个 token，每个 token 都有一个简化 Query/Key
Q_all = [[1,0], [1,1], [0,1], [1,-1]]
K_all = [[1,0], [0.7,0.7], [0,1], [1,-1]]

for i, q in enumerate(Q_all):
    raw = [dot(q, k) for k in K_all]
    # 屏蔽 j > i 的未来位置。用一个非常小的有限数模拟 -infinity。
    masked = [s if j <= i else -1e9 for j, s in enumerate(raw)]
    w = softmax(masked)
    print(f'query token {i}:')
    print('  raw    =', [round(x,3) for x in raw])
    print('  masked =', ['-inf' if x < -1e8 else round(x,3) for x in masked])
    print('  weight =', [round(x,4) for x in w])

print('\n观察：每个位置之后的未来 token 权重都变成 0。')
