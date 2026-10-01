# 第10讲：位置编码玩具实验
# 只依赖 numpy：pip install numpy
import numpy as np
np.set_printoptions(precision=4, suppress=True)

def softmax(x, axis=-1):
    x=x-np.max(x,axis=axis,keepdims=True); e=np.exp(x); return e/e.sum(axis=axis,keepdims=True)

def self_attention(X):
    Q=K=V=X.copy(); scores=Q@K.T/np.sqrt(X.shape[-1]); w=softmax(scores,axis=-1); return w@V,w

print("=== 实验 A：没有位置时的置换等变性 ===")
X=np.array([[1.0,0.0],[0.2,0.9],[-0.8,0.3]])
Y,W=self_attention(X)
perm=np.array([2,1,0]); Xp=X[perm]; Yp,Wp=self_attention(Xp)
print("原输入输出 Y:\n",Y)
print("排列后输出 Yp:\n",Yp)
print("原输出按同样排列 Y[perm]:\n",Y[perm])
print("最大差值:",np.max(np.abs(Yp-Y[perm])))

print("\n=== 实验 B：加入极简绝对位置向量 ===")
P=np.array([[0.0,0.0],[0.3,-0.1],[0.6,-0.2]])
Y_pos,_=self_attention(X+P); Yp_pos,_=self_attention(Xp+P)
print("原序列 + 位置 输出:\n",Y_pos)
print("重排 token 后 + 新位置 输出:\n",Yp_pos)
print("与简单置换的最大差值:",np.max(np.abs(Yp_pos-Y_pos[perm])))

print("\n=== 实验 C：二维 RoPE 旋转直觉 ===")
def rotate(v,theta):
    R=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    return R@v
q=np.array([1.0,0.4]); k=np.array([0.7,0.8]); theta_base=0.35
for m,n in [(1,2),(1,5),(4,5)]:
    qm=rotate(q,m*theta_base); kn=rotate(k,n*theta_base)
    print(f"position m={m}, n={n}, distance={n-m}, dot={qm@kn:.4f}")
print("\n观察：内容 q/k 不变，仅位置变化，就会改变旋转后的点积。")
