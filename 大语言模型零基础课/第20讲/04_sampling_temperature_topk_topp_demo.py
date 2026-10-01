import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(42)

tokens = np.array(["model", "system", "agent", "data", "tool", "banana"])
logits = np.array([4.2, 3.4, 2.8, 2.1, 1.6, 0.3], dtype=float)

def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()

def temperature_probs(logits, T):
    if T <= 0:
        raise ValueError("T must be > 0 for the mathematical temperature formula")
    return softmax(logits / T)

def top_k_filter(probs, k):
    idx = np.argsort(probs)[::-1]
    keep = idx[:k]
    out = np.zeros_like(probs)
    out[keep] = probs[keep]
    return out / out.sum()

def top_p_filter(probs, p):
    idx = np.argsort(probs)[::-1]
    sorted_probs = probs[idx]
    cumulative = np.cumsum(sorted_probs)
    cutoff = np.searchsorted(cumulative, p, side="left")
    keep = idx[:cutoff+1]
    out = np.zeros_like(probs)
    out[keep] = probs[keep]
    return out / out.sum(), keep

print("TOKENS:", tokens.tolist())
print("LOGITS:", logits)

settings = [0.3, 1.0, 2.0]
all_probs = []
for T in settings:
    probs = temperature_probs(logits, T)
    all_probs.append(probs)
    print(f"\nTemperature T={T}")
    for tok, pr in zip(tokens, probs):
        print(f"  {tok:>7s}: {pr:.4f}")

base = temperature_probs(logits, 1.0)
print("\nTop-k with k=3 (after T=1.0)")
k_probs = top_k_filter(base, 3)
for tok, pr in zip(tokens, k_probs):
    print(f"  {tok:>7s}: {pr:.4f}")

print("\nTop-p with p=0.80 (after T=1.0)")
p_probs, keep = top_p_filter(base, 0.80)
print("  kept tokens:", tokens[keep].tolist())
for tok, pr in zip(tokens, p_probs):
    print(f"  {tok:>7s}: {pr:.4f}")

# Sampling frequencies.
def sample_many(probs, n=10000):
    ids = rng.choice(len(tokens), size=n, p=probs)
    counts = np.bincount(ids, minlength=len(tokens))
    return counts / n

print("\nSampling frequencies over 10,000 draws")
for name, probs in [
    ("T=0.3", all_probs[0]),
    ("T=1.0", all_probs[1]),
    ("T=2.0", all_probs[2]),
    ("top-k=3", k_probs),
    ("top-p=0.80", p_probs),
]:
    freq = sample_many(probs)
    print(name, dict(zip(tokens.tolist(), np.round(freq, 4).tolist())))

# Visual comparison: one separate figure.
x = np.arange(len(tokens))
width = 0.24
plt.figure(figsize=(9, 4.8))
plt.bar(x-width, all_probs[0], width, label='T=0.3')
plt.bar(x,       all_probs[1], width, label='T=1.0')
plt.bar(x+width, all_probs[2], width, label='T=2.0')
plt.xticks(x, tokens)
plt.ylabel('Probability')
plt.title('Temperature changes probability sharpness')
plt.legend()
plt.tight_layout()
plt.savefig(r"/mnt/data/LLM_Lesson20_Material_Pack/sampling_probability_demo.png", dpi=180)
print("\nSaved plot:", r"/mnt/data/LLM_Lesson20_Material_Pack/sampling_probability_demo.png")
