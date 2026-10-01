# 第32讲实验：Calibration + ECE + Risk-Coverage + Agent Disagreement
# 教学模拟，不代表任何真实模型。

import random
import math

random.seed(32)

N = 1000

# -------------------------
# A. Synthetic over-confident system
# -------------------------
# latent difficulty determines real correctness probability
samples = []
for _ in range(N):
    true_p = random.uniform(0.45, 0.95)
    correct = 1 if random.random() < true_p else 0

    # system is deliberately over-confident
    reported = min(0.99, 0.15 + 0.95 * true_p)

    samples.append((reported, correct))

def expected_calibration_error(samples, bins=10):
    total = len(samples)
    ece = 0.0
    details = []
    for b in range(bins):
        lo = b / bins
        hi = (b + 1) / bins
        bucket = [(c,y) for c,y in samples if (lo <= c < hi) or (b == bins-1 and c == 1.0)]
        if not bucket:
            continue
        avg_conf = sum(c for c,_ in bucket) / len(bucket)
        acc = sum(y for _,y in bucket) / len(bucket)
        weight = len(bucket) / total
        ece += weight * abs(acc - avg_conf)
        details.append((lo, hi, len(bucket), avg_conf, acc))
    return ece, details

ece, details = expected_calibration_error(samples, bins=10)
accuracy = sum(y for _,y in samples) / N
avg_conf = sum(c for c,_ in samples) / N

print("PART A: Calibration")
print("=" * 72)
print(f"N={N}")
print(f"accuracy={accuracy:.3f}")
print(f"average reported confidence={avg_conf:.3f}")
print(f"ECE={ece:.3f}")
print()
print("Bin details:")
for lo,hi,n,conf,acc in details:
    print(f"[{lo:.1f},{hi:.1f}) n={n:4d} conf={conf:.3f} acc={acc:.3f}")

# -------------------------
# B. Risk-Coverage
# -------------------------
print()
print("PART B: Risk-Coverage")
print("=" * 72)
ranked = sorted(samples, key=lambda x: x[0], reverse=True)
for coverage in [1.0, 0.8, 0.6, 0.4, 0.2]:
    keep = max(1, int(N * coverage))
    subset = ranked[:keep]
    acc = sum(y for _,y in subset) / keep
    risk = 1 - acc
    threshold = subset[-1][0]
    print(
        f"coverage={coverage:.1f} "
        f"threshold>={threshold:.3f} "
        f"selective_accuracy={acc:.3f} risk={risk:.3f}"
    )

# -------------------------
# C. Multi-Agent agreement can be misleading
# -------------------------
print()
print("PART C: Agent Agreement")
print("=" * 72)

def simulate_agent_case(shared_systematic_error=False, trials=5000):
    high_agreement_wrong = 0
    high_agreement_total = 0
    for _ in range(trials):
        if shared_systematic_error and random.random() < 0.25:
            # all 3 agents share the same wrong answer
            answers = [0,0,0]
            truth = 1
        else:
            truth = 1
            answers = [1 if random.random() < 0.70 else 0 for _ in range(3)]

        agreement = max(sum(answers), 3 - sum(answers)) / 3
        if agreement == 1.0:
            high_agreement_total += 1
            majority = 1 if sum(answers) >= 2 else 0
            if majority != truth:
                high_agreement_wrong += 1

    wrong_rate = high_agreement_wrong / high_agreement_total if high_agreement_total else 0.0
    return high_agreement_total, wrong_rate

for shared in [False, True]:
    n_agree, wrong_rate = simulate_agent_case(shared_systematic_error=shared)
    label = "with shared systematic error" if shared else "mostly independent"
    print(f"{label:30s} | unanimous cases={n_agree:4d} | wrong among unanimous={wrong_rate:.3f}")

print()
print("Lesson:")
print("- High confidence is useful only if calibrated.")
print("- Selective prediction trades coverage for lower risk.")
print("- Unanimous agents can still be wrong when they share a systematic error.")
