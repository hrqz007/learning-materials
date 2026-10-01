# -*- coding: utf-8 -*-
"""第17讲：Preference Learning / Reward Model / DPO 教学玩具实验
不依赖第三方库。它不是完整 LLM 训练器，只演示损失函数的方向。
"""
import math

def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)

def pairwise_rm_loss(r_chosen, r_rejected):
    margin = r_chosen - r_rejected
    return -math.log(sigmoid(margin) + 1e-12)

def dpo_loss(logp_c_pi, logp_r_pi, logp_c_ref, logp_r_ref, beta=0.5):
    delta_pi = logp_c_pi - logp_r_pi
    delta_ref = logp_c_ref - logp_r_ref
    advantage = delta_pi - delta_ref
    loss = -math.log(sigmoid(beta * advantage) + 1e-12)
    return delta_pi, delta_ref, advantage, loss

print('=== Experiment A: Pairwise Reward Model Loss ===')
r_rejected = -0.1
for r_chosen in [0.0, 0.2, 0.6, 1.0, 1.4]:
    loss = pairwise_rm_loss(r_chosen, r_rejected)
    print(f'r_chosen={r_chosen:>4.1f}, r_rejected={r_rejected:>4.1f}, margin={r_chosen-r_rejected:>4.1f}, loss={loss:.4f}')

print('\nInterpretation: chosen reward 比 rejected 高得越明显，pairwise loss 越小。')

print('\n=== Experiment B: Simplified DPO-style Loss ===')
# Reference model slightly prefers rejected.
logp_c_ref = -3.2
logp_r_ref = -3.0

cases = [
    ('policy-1: still prefers rejected', -3.0, -2.8),
    ('policy-2: nearly tied',          -2.8, -2.85),
    ('policy-3: prefers chosen',       -2.5, -3.0),
    ('policy-4: strongly chosen',      -2.0, -3.2),
]

for name, logp_c_pi, logp_r_pi in cases:
    dpi, dref, adv, loss = dpo_loss(logp_c_pi, logp_r_pi, logp_c_ref, logp_r_ref, beta=0.5)
    print(f'\n{name}')
    print(f'  delta_policy = {dpi:.3f}')
    print(f'  delta_ref    = {dref:.3f}')
    print(f'  relative advantage = {adv:.3f}')
    print(f'  DPO-style loss     = {loss:.4f}')

print('\nInterpretation: 如果 policy 相对 reference 更偏向 chosen，relative advantage 变大，loss 下降。')
print('\n注意：真实 DPO 会对完整 response 的 token log-prob 求和/聚合，并反向传播到神经网络参数。')
