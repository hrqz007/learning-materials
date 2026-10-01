import numpy as np

np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(7)

d_model = 8
vocab = ["<BOS>", "I", "like", "study", "AI", ".", "<EOS>"]
V = len(vocab)

# Toy embedding and one-head attention weights.
E  = rng.normal(0, 0.5, size=(V, d_model))
Wq = rng.normal(0, 0.4, size=(d_model, d_model))
Wk = rng.normal(0, 0.4, size=(d_model, d_model))
Wv = rng.normal(0, 0.4, size=(d_model, d_model))
Wo = rng.normal(0, 0.4, size=(d_model, V))

prompt_ids = [0, 1, 2]   # <BOS> I like
max_new_tokens = 4

def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()

def full_causal_attention(ids):
    """Naive: recompute Q/K/V for the whole sequence every decode step."""
    X = E[ids]
    Q, K, Vv = X @ Wq, X @ Wk, X @ Wv
    n = len(ids)
    scores = Q @ K.T / np.sqrt(d_model)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    scores[mask] = -1e9
    A = np.zeros_like(scores)
    for i in range(n):
        A[i] = softmax(scores[i])
    H = A @ Vv
    return H, K, Vv

# ---------- Prefill ----------
H, K_cache, V_cache = full_causal_attention(prompt_ids)
last_h = H[-1]
print("PROMPT:", [vocab[i] for i in prompt_ids])
print("PREFILL sequence length:", len(prompt_ids))
print("KV cache rows after prefill:", len(K_cache))
print("last hidden shape:", last_h.shape)

# Count only K/V projection rows as a simple teaching proxy.
naive_kv_projection_rows = len(prompt_ids)
cached_kv_projection_rows = len(prompt_ids)

generated = []
ids = prompt_ids.copy()

for step in range(max_new_tokens):
    # LM head: hidden state -> logits over vocabulary.
    logits = last_h @ Wo
    probs = softmax(logits)

    # Greedy choice for deterministic output. Avoid immediately selecting BOS.
    probs[0] = 0.0
    next_id = int(np.argmax(probs))
    generated.append(next_id)
    ids.append(next_id)

    print(f"\nDECODE step {step+1}")
    print("  next token:", vocab[next_id])
    print("  top probability:", float(probs[next_id]))
    print("  context length now:", len(ids))

    if next_id == vocab.index("<EOS>"):
        break

    # ---- cached decode for the newly generated token only ----
    x_new = E[next_id:next_id+1]
    q_new = x_new @ Wq
    k_new = x_new @ Wk
    v_new = x_new @ Wv
    K_cache = np.vstack([K_cache, k_new])
    V_cache = np.vstack([V_cache, v_new])
    scores = (q_new @ K_cache.T / np.sqrt(d_model)).ravel()
    attn = softmax(scores)
    last_h = attn @ V_cache

    # Teaching counter: cached path projects K/V for only 1 new row;
    # naive path would re-project all rows in the growing sequence.
    cached_kv_projection_rows += 1
    naive_kv_projection_rows += len(ids)

print("\nGENERATED:", [vocab[i] for i in generated])
print("FINAL TOKENS:", [vocab[i] for i in ids])
print("\nTeaching proxy for K/V projection work")
print("  naive recomputation rows:", naive_kv_projection_rows)
print("  with KV cache rows:      ", cached_kv_projection_rows)
print("  ratio naive/cache:       ", round(naive_kv_projection_rows / cached_kv_projection_rows, 2))
print("\nNote: real Transformer inference has many layers/heads and more costs.")
print("This counter only illustrates why cached K/V avoids recomputing past K/V projections.")
