# 第35讲实验：Reproducibility + repeated trials + paired bootstrap + seed
# 教学模拟，不代表真实模型。

import random
import statistics
import math

P_A = 0.70
P_B = 0.73
N_ITEMS = 100
N_EXPERIMENTS = 100
BOOTSTRAP_B = 3000

def one_benchmark(seed):
    rng = random.Random(seed)
    # Shared item difficulty introduces pairing.
    results_a = []
    results_b = []
    for _ in range(N_ITEMS):
        difficulty = rng.uniform(-0.15, 0.15)
        pa = min(0.98, max(0.02, P_A + difficulty))
        pb = min(0.98, max(0.02, P_B + difficulty))
        results_a.append(1 if rng.random() < pa else 0)
        results_b.append(1 if rng.random() < pb else 0)
    return results_a, results_b

def accuracy(xs):
    return sum(xs) / len(xs)

def paired_bootstrap_diff(a, b, seed=999, B=BOOTSTRAP_B):
    rng = random.Random(seed)
    n = len(a)
    diffs = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        da = sum(a[i] for i in idx) / n
        db = sum(b[i] for i in idx) / n
        diffs.append(db - da)
    diffs.sort()
    lo = diffs[int(0.025 * B)]
    hi = diffs[int(0.975 * B)]
    return lo, hi

print("PART A: One experiment can mislead")
print("=" * 78)
a, b = one_benchmark(seed=3)
acc_a, acc_b = accuracy(a), accuracy(b)
print(f"Single benchmark seed=3: A={acc_a:.3f} B={acc_b:.3f} diff(B-A)={acc_b-acc_a:+.3f}")

print()
print("PART B: Repeat many independent benchmarks")
print("=" * 78)
diffs = []
a_wins = 0
b_wins = 0
ties = 0
for seed in range(N_EXPERIMENTS):
    a, b = one_benchmark(seed)
    da = accuracy(a)
    db = accuracy(b)
    diffs.append(db - da)
    if da > db:
        a_wins += 1
    elif db > da:
        b_wins += 1
    else:
        ties += 1

print(f"Mean diff(B-A) over {N_EXPERIMENTS} experiments = {statistics.mean(diffs):+.3f}")
print(f"Std of diff = {statistics.stdev(diffs):.3f}")
print(f"A wins={a_wins}, B wins={b_wins}, ties={ties}")

print()
print("PART C: Paired bootstrap CI on one benchmark")
print("=" * 78)
a, b = one_benchmark(seed=42)
diff = accuracy(b) - accuracy(a)
lo, hi = paired_bootstrap_diff(a, b)
print(f"seed=42 A={accuracy(a):.3f} B={accuracy(b):.3f} diff={diff:+.3f}")
print(f"paired bootstrap 95% CI for diff = [{lo:+.3f}, {hi:+.3f}]")

print()
print("PART D: Seed reproduces a pseudo-random sequence")
print("=" * 78)
def sample_tokens(seed):
    rng = random.Random(seed)
    vocab = ["A","B","C"]
    probs = [0.6,0.3,0.1]
    return rng.choices(vocab, weights=probs, k=12)

print("seed=1:", sample_tokens(1))
print("seed=2:", sample_tokens(2))
print("seed=1 again:", sample_tokens(1))

print()
print("Lesson:")
print("- A better true method can still lose on one finite benchmark.")
print("- Repeated trials estimate variability.")
print("- Paired bootstrap puts uncertainty directly on the difference.")
print("- Seed controls pseudo-random draws, not the entire software/hardware world.")
