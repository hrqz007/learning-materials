import numpy as np

np.set_printoptions(precision=3, suppress=True)

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

# 4 tokens, each token has a tiny 3D representation
X = np.array([
    [1.0, 0.2, 0.1],
    [0.3, 1.0, 0.2],
    [0.2, 0.1, 1.0],
    [0.8, 0.4, 0.3],
])

# For simplicity, use X directly as Q/K/V.
Q = X.copy(); K = X.copy(); V = X.copy()
scores = Q @ K.T / np.sqrt(X.shape[1])

print('Raw attention scores:')
print(scores)

# Encoder-style: every token can see every token
enc_weights = softmax(scores, axis=-1)
print('\nEncoder-style attention (full visibility):')
print(enc_weights)

# Decoder-only / GPT-style: token i cannot see positions j > i
mask = np.triu(np.ones_like(scores, dtype=bool), k=1)
masked_scores = scores.copy()
masked_scores[mask] = -1e9

dec_weights = softmax(masked_scores, axis=-1)
print('\nGPT-style causal mask (1 means future is blocked):')
print(mask.astype(int))
print('\nDecoder-only attention (future weights become 0):')
print(dec_weights)

print('\nNotice row 0: in GPT-style attention, the first token can only attend to itself.')
print('Notice row 3: the last token can attend to all previous tokens.')

# A tiny autoregressive loop: NOT a real language model.
# It only demonstrates "predict -> append -> predict again".
vocab = ['AI', 'can', 'learn', 'patterns', '.']
transition = {
    'AI': 'can',
    'can': 'learn',
    'learn': 'patterns',
    'patterns': '.',
    '.': '.',
}

context = ['AI']
print('\nToy autoregressive generation:')
for step in range(4):
    next_token = transition[context[-1]]
    print(f'step {step+1}: context = {context} -> next = {next_token!r}')
    context.append(next_token)

print('\nFinal sequence:', ' '.join(context))
