# 第15讲配套实验：固定 compute 下的 N-D 权衡（教学玩具模型）
# 重要：下面的 loss 函数是人为构造，只用于理解 compute-optimal 的直觉，
# 不代表 Kaplan / Chinchilla 的真实拟合系数，也不能预测任何真实 LLM。

import math

# 假设总训练 compute 预算（任意单位）
C = 6e20

# 粗略关系：C ≈ 6 * N * D
# N: parameters, D: training tokens

def tokens_from_compute(N):
    return C / (6.0 * N)

# 一个“玩具 loss”：
# - 模型太小 -> capacity penalty 大
# - token 太少 -> data penalty 大
# 参数只是为了演示中间会出现一个平衡点。
def toy_loss(N, D):
    N_b = N / 1e9      # billion params
    D_b = D / 1e9      # billion tokens
    irreducible = 1.20
    capacity_penalty = 0.95 / (N_b ** 0.32)
    data_penalty = 1.60 / (D_b ** 0.28)
    return irreducible + capacity_penalty + data_penalty

candidates_b = [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64]
rows=[]
for nb in candidates_b:
    N=nb*1e9
    D=tokens_from_compute(N)
    L=toy_loss(N,D)
    rows.append((nb,D/1e9,L))

best=min(rows,key=lambda x:x[2])

print('固定训练预算 C = %.2e (toy FLOPs)' % C)
print('粗略关系: C ≈ 6ND')
print('\n%-12s %-18s %-10s' % ('N (B params)','D (B tokens)','toy loss'))
print('-'*44)
for nb,db,L in rows:
    mark='  <-- best in this toy grid' if (nb,db,L)==best else ''
    print('%-12.2f %-18.2f %-10.4f%s' % (nb,db,L,mark))

print('\n解释：')
print('1) N 很小时：虽然能训练很多 token，但模型容量不足。')
print('2) N 很大时：每 token 太贵，固定预算下 D 变少，模型可能 undertrained。')
print('3) 中间通常存在一个资源更均衡的区域。')
print('\n再次强调：最优点完全由这个玩具函数决定，不能用于估计真实模型。')

# 再演示“20 tokens/parameter”只是比例概念
print('\n--- 20 tokens/parameter 的尺度感（只做算术）---')
for nb in [1,7,70]:
    tokens=nb*1e9*20
    print('%2dB params -> 约 %.0fB tokens (若采用 20 tokens/param)' % (nb,tokens/1e9))
