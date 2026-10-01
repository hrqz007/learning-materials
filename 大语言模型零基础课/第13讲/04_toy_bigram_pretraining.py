import numpy as np

np.random.seed(7)
np.set_printoptions(precision=3, suppress=True)

# A tiny corpus. This is NOT a Transformer.
# It is the smallest possible language model: P(next_token | current_token).
corpus = "AI can learn . AI can learn patterns . models can learn patterns ."
tokens = corpus.split()

vocab = sorted(set(tokens))
stoi = {tok: i for i, tok in enumerate(vocab)}
itos = {i: tok for tok, i in stoi.items()}
V = len(vocab)

x = np.array([stoi[t] for t in tokens[:-1]], dtype=int)
y = np.array([stoi[t] for t in tokens[1:]], dtype=int)

# trainable parameters: one row of logits for each current token
W = 0.01 * np.random.randn(V, V)


def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def forward_loss_and_grad(W):
    logits = W[x]                 # [N, V]
    probs = softmax(logits)       # P(next | current)
    N = len(x)
    loss = -np.mean(np.log(probs[np.arange(N), y] + 1e-12))

    # dL/dlogits for softmax + cross entropy
    dlogits = probs.copy()
    dlogits[np.arange(N), y] -= 1.0
    dlogits /= N

    dW = np.zeros_like(W)
    for i, token_id in enumerate(x):
        dW[token_id] += dlogits[i]
    return loss, dW

lr = 1.5
print('Vocabulary:', vocab)
print('Training pairs (current -> next):')
for a, b in zip(tokens[:-1], tokens[1:]):
    print(f'  {a:8s} -> {b}')

print('\nTraining...')
for step in range(301):
    loss, grad = forward_loss_and_grad(W)
    W -= lr * grad
    if step % 50 == 0:
        print(f'step {step:3d} | loss = {loss:.4f}')

print('\nLearned next-token probabilities:')
for tok in vocab:
    row = W[stoi[tok]][None, :]
    p = softmax(row)[0]
    top = np.argsort(-p)[:3]
    parts = [f'{itos[j]}:{p[j]:.3f}' for j in top]
    print(f'  after {tok:8s} -> ' + ', '.join(parts))

# autoregressive sampling from the learned bigram model
context = ['AI']
for _ in range(8):
    cur = context[-1]
    p = softmax(W[stoi[cur]][None, :])[0]
    nxt_id = np.random.choice(V, p=p)
    context.append(itos[nxt_id])

print('\nGenerated sequence:')
print(' '.join(context))

print('\nKey idea: even this tiny model uses the same training spine:')
print('tokens -> logits -> softmax -> cross-entropy -> gradient -> parameter update')
