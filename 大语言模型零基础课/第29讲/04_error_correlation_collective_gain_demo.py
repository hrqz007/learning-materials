# 第29讲实验：错误相关性如何影响 Collective Gain
# 目标：
# - 固定每个 Agent 的边际准确率约为 65%
# - 只改变“共享正确/错误事件”的程度
# - 比较 Majority Vote 随 Agent 数量增长的收益
#
# 这是教学模拟，不代表任何真实模型的概率分布。

import random
import statistics
from itertools import combinations

random.seed(2026)

P_CORRECT = 0.65
TRIALS = 20000
AGENT_COUNTS = [1, 3, 5, 9]

def simulate_group(n_agents, shared_prob, trials=TRIALS):
    """
    shared_prob:
      0.0 -> agent correctness almost independent
      1.0 -> all agents share exactly the same correctness event
    The marginal accuracy of each agent remains close to P_CORRECT.
    """
    group_correct = 0
    agent_correct = [0] * n_agents
    errors = [[] for _ in range(n_agents)]

    for _ in range(trials):
        # With shared_prob, all agents share one common correctness outcome.
        use_shared = random.random() < shared_prob
        if use_shared:
            common_correct = random.random() < P_CORRECT
            correctness = [common_correct] * n_agents
        else:
            correctness = [random.random() < P_CORRECT for _ in range(n_agents)]

        for i, c in enumerate(correctness):
            if c:
                agent_correct[i] += 1
                errors[i].append(0)
            else:
                errors[i].append(1)

        votes_correct = sum(correctness)
        if votes_correct > n_agents / 2:
            group_correct += 1

    indiv_acc = sum(x / trials for x in agent_correct) / n_agents
    group_acc = group_correct / trials

    # Average pairwise error correlation using Pearson correlation of 0/1 error vectors.
    if n_agents == 1:
        avg_corr = 0.0
    else:
        cors = []
        for i, j in combinations(range(n_agents), 2):
            a, b = errors[i], errors[j]
            ma, mb = statistics.mean(a), statistics.mean(b)
            va = sum((x - ma) ** 2 for x in a)
            vb = sum((x - mb) ** 2 for x in b)
            if va == 0 or vb == 0:
                corr = 0.0
            else:
                cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
                corr = cov / (va ** 0.5 * vb ** 0.5)
            cors.append(corr)
        avg_corr = sum(cors) / len(cors)

    return indiv_acc, group_acc, avg_corr

SCENARIOS = [
    ("Independent-ish", 0.0),
    ("Low shared errors", 0.2),
    ("Highly correlated", 0.7),
]

if __name__ == "__main__":
    print("Toy multi-agent collective gain simulation")
    print("=" * 86)
    print(f"Target individual accuracy ~= {P_CORRECT:.2f}")
    print()

    for name, shared in SCENARIOS:
        print(f"SCENARIO: {name} (shared_prob={shared})")
        for n in AGENT_COUNTS:
            indiv, group, corr = simulate_group(n, shared)
            gain = group - indiv
            print(
                f"  N={n:2d} | individual={indiv:.3f} "
                f"| majority={group:.3f} | gain={gain:+.3f} "
                f"| avg_error_corr={corr:.3f}"
            )
        print()

    print("Lesson:")
    print("- Same individual accuracy does NOT imply same collective gain.")
    print("- Majority vote benefits from errors that are not too correlated.")
    print("- More agents cannot fix a shared systematic error.")
