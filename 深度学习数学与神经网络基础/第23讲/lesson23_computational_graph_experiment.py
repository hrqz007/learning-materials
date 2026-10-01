import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

CN_FONT = font_manager.FontProperties(fname="/usr/share/fonts/truetype/arphic-gbsn00lp/gbsn00lp.ttf")
plt.rcParams["axes.unicode_minus"] = False

BASE = Path(__file__).resolve().parent
FIG = BASE / "figures"
FIG.mkdir(exist_ok=True)
OUT = BASE / "第23讲_实验实际输出.txt"

lines = []
def log(s=""):
    print(s)
    lines.append(str(s))

def central_diff(f, x, eps=1e-5):
    return (f(x + eps) - f(x - eps)) / (2 * eps)

log("=== 第23讲：Computational Graph 与局部梯度 ===")
log()

# ------------------------------------------------------------
# 实验1：最小节点规则
# ------------------------------------------------------------
log("[实验1] Add / Multiply / Square / ReLU 的局部 backward 规则")
a, b = 2.0, -3.0
incoming = 4.0  # 后方节点传来的 dL/dz

# add: z=a+b
add_da = incoming * 1.0
add_db = incoming * 1.0
log(f"Add node: z=a+b, a={a}, b={b}, 收到梯度={incoming}")
log(f"  传给a的梯度 = {add_da:.6f}")
log(f"  传给b的梯度 = {add_db:.6f}")

# multiply: z=a*b
mul_da = incoming * b
mul_db = incoming * a
log(f"Multiply node: z=a*b, a={a}, b={b}, 收到梯度={incoming}")
log(f"  传给a的梯度 = incoming*b = {mul_da:.6f}")
log(f"  传给b的梯度 = incoming*a = {mul_db:.6f}")

# square: z=a^2
square_da = incoming * 2*a
log(f"Square node: z=a^2, a={a}, 收到梯度={incoming}")
log(f"  传给a的梯度 = incoming*2a = {square_da:.6f}")

# relu
relu_pos = incoming * (1.0 if a > 0 else 0.0)
relu_neg = incoming * (1.0 if b > 0 else 0.0)
log(f"ReLU node: a={a} -> 梯度 {relu_pos:.6f}; b={b} -> 梯度 {relu_neg:.6f}")
log()

# ------------------------------------------------------------
# 实验2：完整计算图手工 backward
# x,w -> multiply -> +b -> relu -> -target -> square -> Loss
# ------------------------------------------------------------
log("[实验2] 完整计算图：手工 Forward + Backward")
x = 2.0
w = 3.0
bias = -1.0
target = 4.0

# Forward
m = w * x
z = m + bias
a_act = max(0.0, z)
d = a_act - target
L = d ** 2

log(f"Forward: x={x}, w={w}, b={bias}, target={target}")
log(f"  m=w*x = {m:.6f}")
log(f"  z=m+b = {z:.6f}")
log(f"  a=ReLU(z) = {a_act:.6f}")
log(f"  d=a-target = {d:.6f}")
log(f"  L=d^2 = {L:.6f}")

# Backward seed: dL/dL = 1
# square L=d^2
_gL = 1.0
g_d = _gL * (2.0 * d)
# subtract d=a-target
g_a = g_d * 1.0
g_target = g_d * (-1.0)
# relu a=relu(z)
g_z = g_a * (1.0 if z > 0 else 0.0)
# add z=m+b
g_m = g_z * 1.0
g_bias = g_z * 1.0
# multiply m=w*x
g_w = g_m * x
g_x = g_m * w

log("Backward:")
log(f"  dL/dL = {_gL:.6f}")
log(f"  dL/dd = {g_d:.6f}")
log(f"  dL/da = {g_a:.6f}")
log(f"  dL/dtarget = {g_target:.6f}  (target通常不是可训练参数)")
log(f"  dL/dz = {g_z:.6f}")
log(f"  dL/dm = {g_m:.6f}")
log(f"  dL/db = {g_bias:.6f}")
log(f"  dL/dw = {g_w:.6f}")
log(f"  dL/dx = {g_x:.6f}")
log()

# ------------------------------------------------------------
# 实验3：finite-difference gradient check
# ------------------------------------------------------------
log("[实验3] 数值差分检查 w / b / x 梯度")
def loss_value(xv, wv, bv, tv=target):
    mv = wv * xv
    zv = mv + bv
    av = max(0.0, zv)
    dv = av - tv
    return dv ** 2

num_w = central_diff(lambda q: loss_value(x, q, bias), w)
num_b = central_diff(lambda q: loss_value(x, w, q), bias)
num_x = central_diff(lambda q: loss_value(q, w, bias), x)
log(f"  manual dL/dw = {g_w:.6f}, numeric = {num_w:.6f}, abs error = {abs(g_w-num_w):.3e}")
log(f"  manual dL/db = {g_bias:.6f}, numeric = {num_b:.6f}, abs error = {abs(g_bias-num_b):.3e}")
log(f"  manual dL/dx = {g_x:.6f}, numeric = {num_x:.6f}, abs error = {abs(g_x-num_x):.3e}")
log()

# ------------------------------------------------------------
# 实验4：ReLU 关闭路径
# ------------------------------------------------------------
log("[实验4] ReLU 关闭后，这条路径的梯度被截断")
w2 = -3.0
m2 = w2 * x
z2 = m2 + bias
act2 = max(0.0, z2)
d2 = act2 - target
L2 = d2**2
incoming2 = 2*d2
relu_local2 = 1.0 if z2 > 0 else 0.0
gz2 = incoming2 * relu_local2
gw2 = gz2 * x
num_w2 = central_diff(lambda q: loss_value(x, q, bias), w2)
log(f"  w={w2}, z={z2:.6f}, ReLU(z)={act2:.6f}, Loss={L2:.6f}")
log(f"  ReLU local derivative={relu_local2:.1f}")
log(f"  manual dL/dw={gw2:.6f}, numeric={num_w2:.6f}")
log()

# ------------------------------------------------------------
# 实验5：共享变量 / 梯度累加
# L = x^2 + 3x
# ------------------------------------------------------------
log("[实验5] 一个变量被使用两次：梯度必须累加")
x3 = 2.0
u = x3*x3
v = 3.0*x3
L3 = u + v
# backward from add: both paths receive 1
path1 = 1.0 * (2*x3)
path2 = 1.0 * 3.0
gx3 = path1 + path2
num_x3 = central_diff(lambda q: q*q + 3*q, x3)
log(f"  x={x3}, L=x^2+3x={L3:.6f}")
log(f"  path1 contribution (x^2) = {path1:.6f}")
log(f"  path2 contribution (3x) = {path2:.6f}")
log(f"  accumulated dL/dx = {gx3:.6f}, numeric = {num_x3:.6f}")
log()

# ------------------------------------------------------------
# 实验6：一次最小参数更新方向演示
# ------------------------------------------------------------
log("[实验6] 用手工梯度做一次最小参数更新")
lr = 0.05
new_w = w - lr*g_w
new_b = bias - lr*g_bias
new_loss = loss_value(x, new_w, new_b)
log(f"  old w={w:.6f}, old b={bias:.6f}, old Loss={L:.6f}")
log(f"  lr={lr}")
log(f"  new w={new_w:.6f}, new b={new_b:.6f}, new Loss={new_loss:.6f}")
log("  这里还不是完整训练循环，只演示 backward 产生的梯度如何被参数更新使用。")
log()

# ------------------------------------------------------------
# Figure 1: computational graph forward/backward values
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 4.8))
ax.axis('off')
xs = np.linspace(0.05, 0.95, 8)
y0 = 0.58
nodes = [
    ("x", f"{x:.1f}", f"dL/dx={g_x:.1f}"),
    ("w", f"{w:.1f}", f"dL/dw={g_w:.1f}"),
    ("m=w*x", f"{m:.1f}", f"dL/dm={g_m:.1f}"),
    ("z=m+b", f"{z:.1f}", f"dL/dz={g_z:.1f}"),
    ("a=ReLU(z)", f"{a_act:.1f}", f"dL/da={g_a:.1f}"),
    ("d=a-y", f"{d:.1f}", f"dL/dd={g_d:.1f}"),
    ("L=d²", f"{L:.1f}", "seed=1"),
]
# special layout: w joins multiply; b and target not shown as full chain to keep readable
chain_x = [0.08,0.25,0.42,0.59,0.76,0.91]
labels = [
    (chain_x[0],"x=2\nw=3","grads: x=6, w=4"),
    (chain_x[1],"m=w*x=6","grad=2"),
    (chain_x[2],"z=m+b=5\nb=-1","grads: m=2, b=2"),
    (chain_x[3],"a=ReLU(z)=5","grad=2"),
    (chain_x[4],"d=a-y=1\ny=4","grad=2"),
    (chain_x[5],"L=d²=1","seed=1"),
]
for i,(xp,main,grad) in enumerate(labels):
    ax.text(xp,y0,main,ha='center',va='center',fontsize=11,
            bbox=dict(boxstyle='round,pad=0.5',fc='white',ec='black'))
    ax.text(xp,y0-0.24,grad,ha='center',va='center',fontsize=9)
    if i < len(labels)-1:
        ax.annotate('',xy=(labels[i+1][0]-0.055,y0),xytext=(xp+0.055,y0),arrowprops=dict(arrowstyle='->',lw=1.5))
        ax.annotate('',xy=(xp+0.055,y0-0.10),xytext=(labels[i+1][0]-0.055,y0-0.10),arrowprops=dict(arrowstyle='->',lw=1.2,linestyle='--'))
ax.text(0.5,0.93,'Forward：实线从左到右算数值；Backward：虚线从右到左传梯度',ha='center',fontsize=12,fontproperties=CN_FONT)
ax.set_xlim(0,1); ax.set_ylim(0,1)
fig.tight_layout()
fig.savefig(FIG/'lesson23_forward_backward_graph.png',dpi=180,bbox_inches='tight')
plt.close(fig)

# Figure 2: node rule bars
names = ['Add→a','Add→b','Mul→a','Mul→b','Square→a','ReLU(+)', 'ReLU(-)']
vals = [add_da,add_db,mul_da,mul_db,square_da,relu_pos,relu_neg]
fig, ax = plt.subplots(figsize=(9,4.8))
ax.bar(names, vals)
ax.axhline(0,linewidth=1)
ax.set_ylabel('传给输入的梯度', fontproperties=CN_FONT)
ax.set_title('同一个 incoming gradient=4，经过不同节点后的梯度', fontproperties=CN_FONT)
ax.tick_params(axis='x',rotation=25)
fig.tight_layout()
fig.savefig(FIG/'lesson23_local_node_rules.png',dpi=180,bbox_inches='tight')
plt.close(fig)

# Figure 3: manual vs numeric gradient check
manual = np.array([g_w,g_bias,g_x,gx3],dtype=float)
numeric = np.array([num_w,num_b,num_x,num_x3],dtype=float)
labels_gc = ['dL/dw','dL/db','dL/dx','branch dL/dx']
pos = np.arange(len(manual)); width=0.36
fig, ax = plt.subplots(figsize=(8.5,4.8))
ax.bar(pos-width/2,manual,width,label='manual backward')
ax.bar(pos+width/2,numeric,width,label='numeric diff')
ax.set_xticks(pos); ax.set_xticklabels(labels_gc)
ax.set_ylabel('gradient')
ax.set_title('手工 backward 与数值差分一致', fontproperties=CN_FONT)
ax.legend()
fig.tight_layout()
fig.savefig(FIG/'lesson23_gradient_check.png',dpi=180,bbox_inches='tight')
plt.close(fig)

OUT.write_text('\n'.join(lines), encoding='utf-8')
