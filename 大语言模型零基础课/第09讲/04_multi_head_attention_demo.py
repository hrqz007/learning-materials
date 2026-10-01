"""
第9讲：Multi-Head Attention 玩具实验
只用 Python 标准库。
目的：观察两个 Head 产生不同 scores / weights / outputs，
然后 Concat，再通过 W_O 混合。
"""
import math

def dot(a,b):
    return sum(x*y for x,y in zip(a,b))

def softmax(xs):
    m=max(xs)
    exps=[math.exp(x-m) for x in xs]
    s=sum(exps)
    return [e/s for e in exps]

def weighted_sum(weights, values):
    d=len(values[0])
    out=[0.0]*d
    for w,v in zip(weights,values):
        for j in range(d):
            out[j]+=w*v[j]
    return out

def matvec(v, W):
    # v: [din], W: [din][dout]
    return [sum(v[i]*W[i][j] for i in range(len(v))) for j in range(len(W[0]))]

def attention_one_query(Q, K, V):
    scale=math.sqrt(len(Q))
    scores=[dot(Q,k)/scale for k in K]
    weights=softmax(scores)
    out=weighted_sum(weights,V)
    return scores,weights,out

# 同一个“当前位置”，两个 Head 用不同的 Q/K/V 子空间。
head1_Q=[1.0,0.0]
head1_K=[[1.0,0.0],[0.7,0.3],[0.0,1.0]]
head1_V=[[2.0,0.0],[1.0,1.0],[0.0,2.0]]

head2_Q=[0.0,1.0]
head2_K=[[1.0,0.0],[0.2,0.8],[0.0,1.0]]
head2_V=[[10.0,0.0],[0.0,10.0],[5.0,5.0]]

s1,w1,o1=attention_one_query(head1_Q,head1_K,head1_V)
s2,w2,o2=attention_one_query(head2_Q,head2_K,head2_V)

print('=== Head 1 ===')
print('scores :', [round(x,4) for x in s1])
print('weights:', [round(x,4) for x in w1], 'sum=', round(sum(w1),6))
print('output :', [round(x,4) for x in o1])

print('\n=== Head 2 ===')
print('scores :', [round(x,4) for x in s2])
print('weights:', [round(x,4) for x in w2], 'sum=', round(sum(w2),6))
print('output :', [round(x,4) for x in o2])

concat=o1+o2
print('\n=== Concat ===')
print('concat :', [round(x,4) for x in concat])

# 一个 4 -> 4 的玩具 W_O。真实模型参数由训练得到。
W_O=[
    [0.8,0.0,0.2,0.0],
    [0.0,0.8,0.0,0.2],
    [0.2,0.0,0.8,0.0],
    [0.0,0.2,0.0,0.8],
]
final=matvec(concat,W_O)
print('\n=== After W_O ===')
print('final  :', [round(x,4) for x in final])

print('\n观察：两个 Head 的权重和输出不同；Concat 后 W_O 再把不同 Head 的信息混合。')
