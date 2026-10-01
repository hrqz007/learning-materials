# 第11讲：极小 Transformer Block（Pre-Norm）玩具实验
# 只依赖 numpy：pip install numpy
import numpy as np
np.set_printoptions(precision=4, suppress=True)
np.random.seed(7)

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def layer_norm(x, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)

def causal_mask(n):
    m = np.triu(np.ones((n, n)), k=1)
    return np.where(m == 1, -1e9, 0.0)

def mha(x, Wq, Wk, Wv, Wo, n_heads=2):
    n, d_model = x.shape
    d_head = d_model // n_heads
    Q = x @ Wq
    K = x @ Wk
    V = x @ Wv
    # [n, d_model] -> [heads, n, d_head]
    Q = Q.reshape(n, n_heads, d_head).transpose(1, 0, 2)
    K = K.reshape(n, n_heads, d_head).transpose(1, 0, 2)
    V = V.reshape(n, n_heads, d_head).transpose(1, 0, 2)
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_head)
    scores = scores + causal_mask(n)[None, :, :]
    weights = softmax(scores, axis=-1)
    heads = weights @ V
    concat = heads.transpose(1, 0, 2).reshape(n, d_model)
    return concat @ Wo, weights

def ffn(x, W1, W2):
    hidden = x @ W1
    # ReLU for clarity; modern LLM often use GELU/SiLU/SwiGLU
    hidden = np.maximum(hidden, 0.0)
    return hidden @ W2, hidden

# ----- dimensions -----
n = 4
nd_model = 4
d_ff = 8
n_heads = 2

# Four token hidden states
X = np.array([
    [ 0.8,  0.1, -0.2,  0.4],
    [ 0.2,  0.9,  0.3, -0.1],
    [-0.4,  0.3,  0.7,  0.5],
    [ 0.1, -0.2,  0.6,  0.9],
], dtype=float)

scale = 0.35
Wq = np.random.randn(nd_model, nd_model) * scale
Wk = np.random.randn(nd_model, nd_model) * scale
Wv = np.random.randn(nd_model, nd_model) * scale
Wo = np.random.randn(nd_model, nd_model) * scale
W1 = np.random.randn(nd_model, d_ff) * scale
W2 = np.random.randn(d_ff, nd_model) * scale

print('=== Input X ===')
print('shape:', X.shape)
print(X)

# First sublayer: Pre-Norm + MHA + residual
Xn = layer_norm(X)
A_update, attn_w = mha(Xn, Wq, Wk, Wv, Wo, n_heads=n_heads)
A = X + A_update

print('\n=== After Norm(X) ===')
print(Xn)
print('\n=== Attention weights: head 0 ===')
print(attn_w[0])
print('\n=== Attention update ===')
print(A_update)
print('\n=== Residual A = X + AttentionUpdate ===')
print('shape:', A.shape)
print(A)

# Second sublayer: Pre-Norm + FFN + residual
An = layer_norm(A)
F_update, F_hidden = ffn(An, W1, W2)
Y = A + F_update

print('\n=== FFN hidden ===')
print('shape:', F_hidden.shape, '(n, d_ff)')
print(F_hidden)
print('\n=== FFN update ===')
print(F_update)
print('\n=== Final Y = A + FFNUpdate ===')
print('shape:', Y.shape)
print(Y)

print('\nKey observations:')
print('1) X, A, Y all keep shape (n, d_model).')
print('2) Attention mixes information across allowed tokens.')
print('3) FFN expands to d_ff, processes each token, then returns to d_model.')
print('4) Residual additions preserve the main representation stream.')
