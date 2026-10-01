# 第31讲实验：Pass@k + Toy LLM-as-a-Judge Bias
# 目的：
# A. 看清 k 增大为什么会自然抬高“至少一次成功”的概率
# B. 看清 Judge 的 position bias / verbosity bias 怎样改变评测结论
#
# 这是教学模拟，不代表任何真实模型或真实评测器。

import math
import random

random.seed(31)

def theoretical_pass_at_k(p, k):
    return 1 - (1 - p) ** k

def empirical_pass_at_k(n, c, k):
    # Standard finite-sample estimator:
    # 1 - C(n-c, k) / C(n, k)
    if k > n:
        raise ValueError("k cannot exceed n")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)

print("PART A: Pass@k")
print("=" * 64)
p = 0.30
for k in [1, 2, 5, 10]:
    print(f"theoretical p={p:.2f}, k={k:2d} -> Pass@k={theoretical_pass_at_k(p,k):.3f}")

print()
n, c = 20, 6
for k in [1, 2, 5, 10]:
    print(f"empirical n={n}, c={c}, k={k:2d} -> Pass@k={empirical_pass_at_k(n,c,k):.3f}")

print()
print("PART B: Toy Judge Bias")
print("=" * 64)

answer_short = "Use exponential backoff for HTTP 429 and respect Retry-After."
answer_long = (
    "When an HTTP 429 response occurs, the client should use exponential backoff, "
    "respect the Retry-After header when provided, avoid immediate tight-loop retries, "
    "and log repeated rate-limit events for later diagnosis."
)

# Assume both are factually acceptable for this toy example.
# Judge score includes:
# base quality + verbosity bonus + position bonus.
def toy_judge(answer_a, answer_b, position_bias=0.08, verbosity_bias=0.002):
    def score(text, is_a):
        base = 1.0
        verbosity = verbosity_bias * len(text.split())
        position = position_bias if is_a else 0.0
        return base + verbosity + position
    sa = score(answer_a, True)
    sb = score(answer_b, False)
    if abs(sa - sb) < 1e-9:
        winner = "Tie"
    else:
        winner = "A" if sa > sb else "B"
    return sa, sb, winner

sa, sb, winner = toy_judge(answer_short, answer_long)
print("Order 1: A=short, B=long")
print(f"score(A)={sa:.3f} score(B)={sb:.3f} winner={winner}")

sa2, sb2, winner2 = toy_judge(answer_long, answer_short)
print("Order 2: A=long, B=short")
print(f"score(A)={sa2:.3f} score(B)={sb2:.3f} winner={winner2}")

print()
print("Lesson:")
print("- Pass@k rises when you allow more attempts; this is extra test-time budget.")
print("- A judge can prefer position or verbosity even when factual quality is equal.")
print("- Evaluation protocol can change the apparent winner.")
