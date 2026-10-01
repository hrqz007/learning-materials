# 第24讲实验：语言先验 vs 外部证据（Grounding）
# 使用虚构实体，演示“语言先验 != 事实真值”。

import random
from collections import Counter

random.seed(7)

FACTS = {
    "Lyria 的首都是？": "Neris",
    "Orven 的货币是？": "Solin",
    "Talma 的国鸟是？": "Varek",
    "Edria 的最大湖泊是？": "Mira",
}

PRIORS = {
    "Lyria 的首都是？": {"Valora": 0.55, "Neris": 0.25, "Candon": 0.20},
    "Orven 的货币是？": {"Crown": 0.50, "Solin": 0.30, "Daro": 0.20},
    "Talma 的国鸟是？": {"Falcon": 0.60, "Varek": 0.25, "Rin": 0.15},
    "Edria 的最大湖泊是？": {"Aster": 0.50, "Mira": 0.35, "Lune": 0.15},
}

def sample_from_dist(dist):
    names = list(dist)
    probs = [dist[n] for n in names]
    return random.choices(names, probs, k=1)[0]

def prior_only(question):
    return sample_from_dist(PRIORS[question])

def retrieve(question, noise=0.0):
    if random.random() >= noise:
        return FACTS[question]
    wrong = [x for x in PRIORS[question] if x != FACTS[question]]
    return random.choice(wrong)

def grounded(question, noise=0.0):
    evidence = retrieve(question, noise=noise)
    return evidence

def evaluate(method, trials=2000):
    correct = 0
    questions = list(FACTS)
    for _ in range(trials):
        q = random.choice(questions)
        pred = method(q)
        correct += (pred == FACTS[q])
    return correct / trials

if __name__ == "__main__":
    strategies = [
        ("Prior-only", lambda q: prior_only(q)),
        ("Retrieval-grounded", lambda q: grounded(q, noise=0.0)),
        ("Noisy retrieval (15%)", lambda q: grounded(q, noise=0.15)),
        ("Noisy retrieval (30%)", lambda q: grounded(q, noise=0.30)),
    ]
    print("Toy closed-world hallucination / grounding demo")
    print("=" * 56)
    for name, fn in strategies:
        print(f"{name:28s} accuracy = {evaluate(fn):.3f}")
    print()
    q = "Lyria 的首都是？"
    print("Question:", q)
    print("Ground truth:", FACTS[q])
    print("Language prior distribution:", PRIORS[q])
    print("Prior-only sample:", prior_only(q))
    print("Grounded answer:", grounded(q, noise=0.0))
