# 第36讲实验：从 Research Question 到 Paper Table
# 教学目标：
# 1. 把“多 Agent 可能因错误去相关而获得集体收益”变成可测实验。
# 2. 同时报告 Outcome / Mechanism / Cost。
# 3. 展示 repeated runs + paired-style comparison 的基本习惯。
#
# 注意：这是机制教学模拟，不代表真实 LLM 或真实 Agent。

import random
import statistics
import math
from itertools import combinations

N_ITEMS = 900
N_RUNS = 30

# 三类任务：每类 1/3。
TASK_TYPES = [0, 1, 2]

def pearson_binary(xs, ys):
    mx = statistics.mean(xs)
    my = statistics.mean(ys)
    vx = sum((x-mx)**2 for x in xs)
    vy = sum((y-my)**2 for y in ys)
    if vx == 0 or vy == 0:
        return 0.0
    cov = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    return cov / math.sqrt(vx*vy)

def majority(bits):
    return 1 if sum(bits) >= 2 else 0

def simulate_run(seed, mode):
    rng = random.Random(seed)

    # correctness[t][agent] stores 0/1 correctness.
    correctness = []
    calls_per_item = 1 if mode == "single" else 3
    tokens_per_call = 300

    for i in range(N_ITEMS):
        task = TASK_TYPES[i % 3]

        if mode == "single":
            # A single generalist: 70% on all task types.
            row = [1 if rng.random() < 0.70 else 0]

        elif mode == "self_consistency":
            # Same model sampled 3 times.
            # Shared item difficulty -> correlated errors even when random draws are independent.
            difficulty = rng.uniform(-0.18, 0.18)
            p = min(0.95, max(0.20, 0.70 + difficulty))
            row = [1 if rng.random() < p else 0 for _ in range(3)]

        elif mode == "homogeneous_multi":
            # Three homogeneous agents with stronger shared failure events.
            if rng.random() < 0.42:
                common = 1 if rng.random() < 0.70 else 0
                row = [common, common, common]
            else:
                row = [1 if rng.random() < 0.70 else 0 for _ in range(3)]

        elif mode == "experience_diverse":
            # Each agent specializes on one task type.
            # Average individual accuracy is still about 70%:
            # (0.88 + 0.61 + 0.61)/3 = 0.70
            ps = []
            for agent in range(3):
                ps.append(0.88 if agent == task else 0.61)
            row = [1 if rng.random() < p else 0 for p in ps]

        else:
            raise ValueError(mode)

        correctness.append(row)

    if mode == "single":
        group_answers = [row[0] for row in correctness]
        indiv_acc = sum(group_answers)/N_ITEMS
        group_acc = indiv_acc
        corr = 0.0
    else:
        per_agent = list(zip(*correctness))
        indiv_acc = sum(sum(a)/N_ITEMS for a in per_agent)/3
        group_answers = [majority(row) for row in correctness]
        group_acc = sum(group_answers)/N_ITEMS
        cors = []
        for i,j in combinations(range(3),2):
            err_i = [1-x for x in per_agent[i]]
            err_j = [1-x for x in per_agent[j]]
            cors.append(pearson_binary(err_i, err_j))
        corr = sum(cors)/len(cors)

    total_calls = calls_per_item * N_ITEMS
    total_tokens = total_calls * tokens_per_call

    return {
        "individual_accuracy": indiv_acc,
        "system_accuracy": group_acc,
        "collective_gain": group_acc - indiv_acc,
        "error_correlation": corr,
        "calls": total_calls,
        "tokens": total_tokens,
    }

MODES = [
    ("single", "Single Agent"),
    ("self_consistency", "Single Agent Self-Consistency x3"),
    ("homogeneous_multi", "Homogeneous Multi-Agent x3"),
    ("experience_diverse", "Experience-Diverse Multi-Agent x3"),
]

all_results = {}
for key, label in MODES:
    runs = [simulate_run(1000+r, key) for r in range(N_RUNS)]
    all_results[key] = runs

def mean_ci(values):
    m = statistics.mean(values)
    if len(values) == 1:
        return m, m, m
    sd = statistics.stdev(values)
    se = sd / math.sqrt(len(values))
    return m, m - 1.96*se, m + 1.96*se

print("Toy research workflow: Outcome + Mechanism + Cost")
print("="*104)
print(f"Items/run={N_ITEMS}, repeated runs={N_RUNS}")
print()

for key, label in MODES:
    runs = all_results[key]
    accs = [r["system_accuracy"] for r in runs]
    indiv = [r["individual_accuracy"] for r in runs]
    gains = [r["collective_gain"] for r in runs]
    corrs = [r["error_correlation"] for r in runs]
    m,lo,hi = mean_ci(accs)
    print(label)
    print(f"  system accuracy     = {m:.3f}  95%CI [{lo:.3f}, {hi:.3f}]")
    print(f"  individual accuracy = {statistics.mean(indiv):.3f}")
    print(f"  collective gain     = {statistics.mean(gains):+.3f}")
    print(f"  error correlation   = {statistics.mean(corrs):.3f}")
    print(f"  calls/run           = {runs[0]['calls']}")
    print(f"  tokens/run          = {runs[0]['tokens']}")
    print()

print("Mechanism question:")
print("Does lower error correlation predict larger collective gain at matched 3-call compute?")
paired_modes = ["self_consistency","homogeneous_multi","experience_diverse"]
pairs = []
for key in paired_modes:
    for r in all_results[key]:
        pairs.append((r["error_correlation"], r["collective_gain"]))

mx = statistics.mean(x for x,_ in pairs)
my = statistics.mean(y for _,y in pairs)
cov = sum((x-mx)*(y-my) for x,y in pairs)
vx = sum((x-mx)**2 for x,_ in pairs)
vy = sum((y-my)**2 for _,y in pairs)
corr_gain = cov / math.sqrt(vx*vy)

print(f"Correlation(error_correlation, collective_gain) = {corr_gain:.3f}")
print()
print("Lesson:")
print("- Outcome alone is not enough.")
print("- Compare same-compute baselines.")
print("- Measure the proposed mechanism directly.")
print("- Repeat runs and report uncertainty.")
