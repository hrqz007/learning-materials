# 第14讲配套实验：一个极小的训练数据管道
# 目标：观察过滤、去重、Data Mixture 与 benchmark contamination 检查。
# 仅使用 Python 标准库。

import re
import random
from collections import Counter, defaultdict

random.seed(14)

docs = [
    {"source":"web", "text":"Python 入门教程：变量用于保存数据，函数用于封装可复用逻辑。"},
    {"source":"web", "text":"Python 入门教程：变量用于保存数据，函数用于封装可复用逻辑。"},  # exact duplicate
    {"source":"web", "text":"Python 入门教程 - 变量用于保存数据，函数用于封装可复用逻辑！"}, # near duplicate
    {"source":"web", "text":"点击这里领取优惠！！！广告 广告 广告"},
    {"source":"book", "text":"神经网络通过可学习参数把输入映射到输出，训练过程通常依赖梯度下降。"},
    {"source":"book", "text":"Transformer 使用注意力机制在序列不同位置之间交换信息。"},
    {"source":"code", "text":"def add(a, b):\n    return a + b"},
    {"source":"code", "text":"def mean(xs):\n    return sum(xs) / len(xs)"},
    {"source":"math", "text":"问题：若 x=3，则 2x+1 等于多少？答案：7。"},
    {"source":"math", "text":"问题：若 x=5，则 2x+1 等于多少？答案：11。"},
    {"source":"benchmark", "text":"测试题：法国的首都是哪里？答案：巴黎。"},
]

benchmark_query = "法国的首都是哪里"

def normalize(s):
    s = s.lower()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[，。！？：；,.!?:;\-]", "", s)
    return s

def quality_ok(doc):
    t = doc["text"]
    if len(t) < 12:
        return False
    bad = ["广告广告广告", "点击这里领取优惠"]
    nt = normalize(t)
    return not any(normalize(x) in nt for x in bad)

def char_ngrams(s, n=4):
    s = normalize(s)
    return {s[i:i+n] for i in range(max(0, len(s)-n+1))}

def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))

def deduplicate(items, threshold=0.78):
    kept=[]
    removed=[]
    for d in items:
        ng = char_ngrams(d["text"])
        dup_of=None
        for k in kept:
            if jaccard(ng, char_ngrams(k["text"])) >= threshold:
                dup_of=k
                break
        if dup_of is None:
            kept.append(d)
        else:
            removed.append((d, dup_of))
    return kept, removed

def stats(label, items):
    c=Counter(d["source"] for d in items)
    chars=sum(len(d["text"]) for d in items)
    print(f"\n[{label}]")
    print("documents:", len(items), " characters:", chars)
    print("sources:", dict(c))

stats("raw", docs)
filtered=[d for d in docs if quality_ok(d)]
stats("after quality filter", filtered)
kept, removed=deduplicate(filtered)
stats("after dedup", kept)
print("\nremoved duplicates:")
for d,k in removed:
    print(" -", d["text"][:28], "...  ~=  ", k["text"][:28], "...")

# Data Mixture: 教学示例，不代表真实模型
mixture={"web":0.30, "book":0.30, "code":0.20, "math":0.20}
pools=defaultdict(list)
for d in kept:
    if d["source"] in mixture:
        pools[d["source"]].append(d)

print("\n[mixture sampling: 20 draws]")
draws=[]
for _ in range(20):
    r=random.random(); acc=0
    chosen=None
    for src,p in mixture.items():
        acc += p
        if r <= acc:
            chosen=src; break
    if pools[chosen]:
        draws.append(random.choice(pools[chosen]))
print(dict(Counter(d["source"] for d in draws)))

# Benchmark contamination check
q=normalize(benchmark_query)
hits=[]
for d in kept:
    nt=normalize(d["text"])
    if q in nt:
        hits.append(d)
print("\n[benchmark contamination check]")
print("query:", benchmark_query)
print("hits:", len(hits))
for h in hits:
    print(" ->", h["source"], h["text"])

print("\n结论：训练数据管道中的过滤、去重、配比和污染检查，会改变模型实际看到的数据分布。")
