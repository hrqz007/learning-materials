# 第25讲实验：一个极小的 Agent Memory Store
# 不依赖第三方库，只演示：
# 1) write
# 2) similarity + recency + importance + reliability 的多信号 retrieval
# 3) 低可信/过期记忆为何可能被降权
#
# 注意：这里的检索公式只是教学示例，不是标准算法。

from dataclasses import dataclass
import math
import re

NOW = 100  # toy time

@dataclass
class Memory:
    text: str
    timestamp: int
    importance: float
    reliability: float

def tokens(text):
    # 简单按英文/数字 token 化，中文示例也可整体作为字符串的一部分
    return set(re.findall(r"[A-Za-z0-9_]+", text.lower()))

def similarity(query, text):
    a, b = tokens(query), tokens(text)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)  # toy Jaccard similarity

def recency(timestamp, now=NOW, half_life=30.0):
    age = max(0, now - timestamp)
    return 0.5 ** (age / half_life)

def score(query, m):
    sim = similarity(query, m.text)
    rec = recency(m.timestamp)
    return (
        0.55 * sim +
        0.20 * rec +
        0.15 * m.importance +
        0.10 * m.reliability
    ), sim, rec

memories = [
    Memory("API timeout happened; verified fix: set timeout 30 and retry once.",
           timestamp=96, importance=0.9, reliability=0.98),
    Memory("Old note: timeout should be 3 seconds.",
           timestamp=35, importance=0.6, reliability=0.25),
    Memory("Database connection failed because password expired.",
           timestamp=90, importance=0.7, reliability=0.95),
    Memory("API rate limit 429; use exponential backoff.",
           timestamp=88, importance=0.8, reliability=0.95),
    Memory("Temporary guess: timeout may be caused by DNS.",
           timestamp=97, importance=0.4, reliability=0.30),
]

query = "How should I handle API timeout?"

ranked = []
for m in memories:
    s, sim, rec = score(query, m)
    ranked.append((s, sim, rec, m))
ranked.sort(reverse=True, key=lambda x: x[0])

print("Query:", query)
print("=" * 78)
for i, (s, sim, rec, m) in enumerate(ranked, 1):
    print(f"{i}. score={s:.3f}  sim={sim:.3f}  recency={rec:.3f}  "
          f"importance={m.importance:.2f}  reliability={m.reliability:.2f}")
    print("   ", m.text)

print()
print("Top memory to inject into context:")
print(ranked[0][3].text)

print()
print("Lesson:")
print("- Similarity alone is not enough.")
print("- Old or unreliable memories can still look semantically relevant.")
print("- A memory system needs write/retrieve/update/forget policies.")
