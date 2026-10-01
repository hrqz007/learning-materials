import numpy as np
import matplotlib.pyplot as plt

# A tiny teaching simulator. One "word" here is treated as one toy token.
WINDOW = 24
OUTPUT_RESERVE = 6
INPUT_BUDGET = WINDOW - OUTPUT_RESERVE

messages = [
    ("system", "You are a careful research assistant"),
    ("user", "My project studies reliable agent systems"),
    ("assistant", "I will keep reliability as the main goal"),
    ("user", "Compare single agent and multi agent baselines"),
    ("assistant", "Use equal compute and repeated random seeds"),
    ("tool", "benchmark result accuracy 0.72 cost 18 calls"),
    ("user", "Now summarize the most important experimental rule"),
]

def toks(text):
    return text.split()

def flatten(msgs):
    items=[]
    for role,text in msgs:
        for t in toks(text):
            items.append((role,t))
    return items

def trim_to_budget(items, budget):
    # A simple sliding-window policy: keep the newest tokens.
    return items[-budget:]

all_items=[]
print(f"CONTEXT WINDOW = {WINDOW} toy tokens")
print(f"RESERVED FOR OUTPUT = {OUTPUT_RESERVE}")
print(f"MAX INPUT TOKENS = {INPUT_BUDGET}\n")

history_lengths=[]
kept_lengths=[]
for turn,(role,text) in enumerate(messages, start=1):
    all_items.extend((role,t) for t in toks(text))
    kept=trim_to_budget(all_items, INPUT_BUDGET)
    history_lengths.append(len(all_items))
    kept_lengths.append(len(kept))
    dropped=max(0,len(all_items)-len(kept))
    print(f"TURN {turn}: role={role}")
    print(f"  total history tokens: {len(all_items)}")
    print(f"  kept in current prompt: {len(kept)}")
    print(f"  dropped from oldest history: {dropped}")
    print("  visible context:", " ".join(tok for _,tok in kept))
    print()

# Demonstrate retrieval as an alternative to stuffing all old text into context.
memory = [
    "project reliable agent systems",
    "baseline equal compute",
    "report multiple random seeds",
    "unrelated cafeteria menu",
    "tool result accuracy 0.72 cost 18 calls",
]
query = set("important experimental rule equal compute random seeds".split())

def overlap_score(text):
    return len(query & set(text.split()))

ranked=sorted(((overlap_score(m),m) for m in memory), reverse=True)
print("RETRIEVAL DEMO")
for score,item in ranked:
    print(f"  score={score}: {item}")
print("Top retrieved memory:", ranked[0][1])

# Plot total conversation tokens vs tokens that actually fit into the input budget.
turns=np.arange(1,len(messages)+1)
plt.figure(figsize=(8.5,4.8))
plt.plot(turns, history_lengths, marker='o', label='Total conversation history')
plt.plot(turns, kept_lengths, marker='o', label='Tokens kept in current context')
plt.axhline(INPUT_BUDGET, linestyle='--', label='Input budget')
plt.xlabel('Turn')
plt.ylabel('Toy token count')
plt.title('Context window: history can grow beyond what the model sees now')
plt.xticks(turns)
plt.legend()
plt.tight_layout()
plt.savefig(r"/mnt/data/LLM_Lesson21_Material_Pack/context_budget_demo.png", dpi=180)
print("\nSaved plot:", r"/mnt/data/LLM_Lesson21_Material_Pack/context_budget_demo.png")
