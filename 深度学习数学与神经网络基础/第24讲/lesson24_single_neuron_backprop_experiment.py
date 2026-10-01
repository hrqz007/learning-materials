from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

CN_FONT = font_manager.FontProperties(fname='/usr/share/fonts/truetype/arphic-gbsn00lp/gbsn00lp.ttf')
plt.rcParams['axes.unicode_minus'] = False

BASE = Path(__file__).resolve().parent
FIG = BASE / 'figures'
FIG.mkdir(exist_ok=True)
OUT = BASE / '第24讲_实验实际输出.txt'

lines=[]
def log(s=''):
    print(s); lines.append(str(s))

def relu(z):
    return np.maximum(0.0, z)

def central_diff_scalar(f, x, eps=1e-5):
    return (f(x+eps)-f(x-eps))/(2*eps)

def loss_of(x, w, b, target):
    z = float(np.dot(x, w) + b)
    a = max(0.0, z)
    return (a-target)**2

log('=== 第24讲：一个神经元的 Backpropagation ===')
log()

# ------------------------------------------------------------
# 实验1：向量输入单神经元完整 Forward + Backward
# ------------------------------------------------------------
log('[实验1] 向量输入单神经元：手工 Forward + Backward')
x = np.array([2.0, -1.0, 3.0], dtype=float)
w = np.array([0.5, -2.0, 1.0], dtype=float)
b = 0.2
target = 5.0

z = float(np.dot(x, w) + b)
a = float(max(0.0, z))
err = a - target
L = err**2

# backward
g_L = 1.0
g_err = g_L * 2.0 * err
g_a = g_err
g_z = g_a * (1.0 if z > 0 else 0.0)
g_w = g_z * x
g_b = g_z
g_x = g_z * w

log(f'x = {x}')
log(f'w = {w}')
log(f'b = {b:.6f}, target = {target:.6f}')
log(f'z = x·w + b = {z:.6f}')
log(f'a = ReLU(z) = {a:.6f}')
log(f'error = a-target = {err:.6f}')
log(f'Loss = error^2 = {L:.6f}')
log('Backward:')
log(f'  dL/da = {g_a:.6f}')
log(f'  da/dz = {1.0 if z>0 else 0.0:.1f}')
log(f'  dL/dz = {g_z:.6f}')
log(f'  dL/dw = {g_w}')
log(f'  dL/db = {g_b:.6f}')
log(f'  dL/dx = {g_x}')
log(f'  shapes: x{tuple(x.shape)}, w{tuple(w.shape)}, dL/dw{tuple(g_w.shape)}, dL/dx{tuple(g_x.shape)}')
log()

# ------------------------------------------------------------
# 实验2：Gradient Check：逐元素验证 w、b、x
# ------------------------------------------------------------
log('[实验2] 数值差分 Gradient Check')
num_w = np.zeros_like(w)
for i in range(len(w)):
    def f(q, i=i):
        w2 = w.copy(); w2[i] = q
        return loss_of(x, w2, b, target)
    num_w[i] = central_diff_scalar(f, w[i])
num_b = central_diff_scalar(lambda q: loss_of(x, w, q, target), b)
num_x = np.zeros_like(x)
for i in range(len(x)):
    def f(q, i=i):
        x2 = x.copy(); x2[i] = q
        return loss_of(x2, w, b, target)
    num_x[i] = central_diff_scalar(f, x[i])

for i in range(len(w)):
    log(f'  w[{i}]: manual={g_w[i]:.6f}, numeric={num_w[i]:.6f}, abs_err={abs(g_w[i]-num_w[i]):.3e}')
log(f'  b:    manual={g_b:.6f}, numeric={num_b:.6f}, abs_err={abs(g_b-num_b):.3e}')
for i in range(len(x)):
    log(f'  x[{i}]: manual={g_x[i]:.6f}, numeric={num_x[i]:.6f}, abs_err={abs(g_x[i]-num_x[i]):.3e}')
log()

# ------------------------------------------------------------
# 实验3：每个权重梯度为什么等于 g_z * 对应输入
# ------------------------------------------------------------
log('[实验3] 权重梯度与输入值的关系')
for i in range(len(w)):
    log(f'  dL/dw[{i}] = dL/dz * x[{i}] = {g_z:.6f} * {x[i]:.6f} = {g_w[i]:.6f}')
log('  直觉：同一个上游梯度 dL/dz，会被各输入 x_i 缩放后分配给对应 w_i。')
log()

# ------------------------------------------------------------
# 实验4：一步参数更新
# ------------------------------------------------------------
log('[实验4] 用梯度做一次最小参数更新')
lr = 0.02
w_new = w - lr * g_w
b_new = b - lr * g_b
L_new = loss_of(x, w_new, b_new, target)
log(f'  lr = {lr}')
log(f'  old w = {w}, old b = {b:.6f}, old Loss = {L:.6f}')
log(f'  new w = {w_new}, new b = {b_new:.6f}, new Loss = {L_new:.6f}')
log()

# ------------------------------------------------------------
# 实验5：ReLU 关闭时，整个神经元前向仍有Loss但前面梯度为0
# ------------------------------------------------------------
log('[实验5] ReLU 关闭：参数梯度为什么会全部归零')
w_off = np.array([-1.0, 1.0, -1.0])
b_off = 0.0
target_off = 2.0
z_off = float(np.dot(x, w_off) + b_off)
a_off = max(0.0, z_off)
err_off = a_off-target_off
L_off = err_off**2
gz_off = (2*err_off) * (1.0 if z_off > 0 else 0.0)
gw_off = gz_off*x
gb_off = gz_off
log(f'  z={z_off:.6f}, a={a_off:.6f}, Loss={L_off:.6f}')
log(f'  dL/dz={gz_off:.6f}')
log(f'  dL/dw={gw_off}, dL/db={gb_off:.6f}')
log()

# ------------------------------------------------------------
# 实验6：PyTorch Autograd 三重对照（可选验证）
# ------------------------------------------------------------
log('[实验6] PyTorch Autograd 对照（可选，不要求现在记语法）')
try:
    import torch
    xt = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    wt = torch.tensor(w, dtype=torch.float64, requires_grad=True)
    bt = torch.tensor(b, dtype=torch.float64, requires_grad=True)
    tt = torch.tensor(target, dtype=torch.float64)
    zt = (xt * wt).sum() + bt
    at = torch.relu(zt)
    Lt = (at-tt)**2
    Lt.backward()
    log(f'  torch Loss = {Lt.item():.6f}')
    log(f'  torch dL/dw = {wt.grad.detach().numpy()}')
    log(f'  torch dL/db = {bt.grad.item():.6f}')
    log(f'  torch dL/dx = {xt.grad.detach().numpy()}')
    log(f'  max |manual-torch| for w = {np.max(np.abs(g_w-wt.grad.detach().numpy())):.3e}')
except Exception as e:
    log(f'  PyTorch 未运行：{type(e).__name__}: {e}')
log()

# ------------------------------------------------------------
# 实验7：Batch 共享权重梯度 = 各样本贡献的聚合（预览）
# ------------------------------------------------------------
log('[实验7] Batch 梯度聚合预览')
X = np.array([[2.0,-1.0,3.0],[1.0,2.0,0.0]])
y = np.array([5.0,-1.0])
zs = X @ w + b
acts = np.maximum(0.0, zs)
errs = acts - y
per_loss = errs**2
mean_loss = per_loss.mean()
# mean reduction -> upstream per sample = 2*err/B
B = X.shape[0]
gz_batch = (2*errs/B) * (zs>0).astype(float)
gw_batch = X.T @ gz_batch
gb_batch = gz_batch.sum()
log(f'  zs = {zs}')
log(f'  per-sample loss = {per_loss}, mean loss = {mean_loss:.6f}')
log(f'  per-sample dL/dz contributions = {gz_batch}')
log(f'  mean-loss dL/dw = {gw_batch}')
log(f'  mean-loss dL/db = {gb_batch:.6f}')
log('  提示：第25讲会把这种向量/矩阵形式扩展到两层 MLP。')

# ---------------- figures ----------------
# Figure 1: neuron forward/backward diagram
fig, ax = plt.subplots(figsize=(11,5.2)); ax.axis('off')
xs=[0.08,0.28,0.48,0.68,0.88]; y0=0.62
labels=[
    ('x,w,b',f'x={x.tolist()}\nw={w.tolist()}\nb={b}',f'dL/dx={np.round(g_x,2).tolist()}\ndL/dw={np.round(g_w,2).tolist()}\ndL/db={g_b:.2f}'),
    ('z=x·w+b',f'z={z:.2f}',f'dL/dz={g_z:.2f}'),
    ('a=ReLU(z)',f'a={a:.2f}',f'dL/da={g_a:.2f}'),
    ('e=a-y',f'e={err:.2f}',f'dL/de={g_err:.2f}'),
    ('L=e²',f'L={L:.2f}','seed=1')
]
for i,(title,val,grad) in enumerate(labels):
    xp=xs[i]
    ax.text(xp,y0,f'{title}\n{val}',ha='center',va='center',fontsize=10,bbox=dict(boxstyle='round,pad=0.45',fc='white',ec='black'))
    ax.text(xp,y0-0.28,grad,ha='center',va='center',fontsize=8.5)
    if i<len(labels)-1:
        ax.annotate('',xy=(xs[i+1]-0.07,y0),xytext=(xp+0.07,y0),arrowprops=dict(arrowstyle='->',lw=1.5))
        ax.annotate('',xy=(xp+0.07,y0-0.1),xytext=(xs[i+1]-0.07,y0-0.1),arrowprops=dict(arrowstyle='->',lw=1.1,linestyle='--'))
ax.text(0.5,0.95,'单神经元：Forward 算数值，Backward 反向传梯度',ha='center',fontsize=13,fontproperties=CN_FONT)
ax.set_xlim(0,1); ax.set_ylim(0,1); fig.tight_layout()
fig.savefig(FIG/'lesson24_single_neuron_forward_backward.png',dpi=180,bbox_inches='tight'); plt.close(fig)

# Figure 2: gradient check
labels_gc=['w0','w1','w2','b','x0','x1','x2']
manual=np.concatenate([g_w,[g_b],g_x]); numeric=np.concatenate([num_w,[num_b],num_x])
pos=np.arange(len(labels_gc)); width=0.36
fig, ax=plt.subplots(figsize=(9,4.8)); ax.bar(pos-width/2,manual,width,label='manual'); ax.bar(pos+width/2,numeric,width,label='numeric diff')
ax.axhline(0,linewidth=1); ax.set_xticks(pos); ax.set_xticklabels(labels_gc); ax.set_ylabel('gradient'); ax.set_title('手工 Backprop 与数值差分一致',fontproperties=CN_FONT); ax.legend(); fig.tight_layout()
fig.savefig(FIG/'lesson24_gradient_check.png',dpi=180,bbox_inches='tight'); plt.close(fig)

# Figure 3: loss before/after update
fig, ax=plt.subplots(figsize=(6.5,4.8)); ax.bar(['更新前','更新后'],[L,L_new]); ax.set_ylabel('Loss'); ax.set_title('沿负梯度做一次小步更新后 Loss 下降',fontproperties=CN_FONT); ax.set_xticklabels(['更新前','更新后'], fontproperties=CN_FONT)
for i,v in enumerate([L,L_new]): ax.text(i,v+0.03,f'{v:.4f}',ha='center')
fig.tight_layout(); fig.savefig(FIG/'lesson24_one_step_update.png',dpi=180,bbox_inches='tight'); plt.close(fig)

# Figure 4: active vs inactive ReLU gradient norm
active_norm=np.linalg.norm(g_w); inactive_norm=np.linalg.norm(gw_off)
fig, ax=plt.subplots(figsize=(6.5,4.8)); ax.bar(['ReLU开启','ReLU关闭'],[active_norm,inactive_norm]); ax.set_ylabel('||dL/dw||'); ax.set_title('ReLU gate 对参数梯度的影响',fontproperties=CN_FONT); ax.set_xticklabels(['ReLU开启','ReLU关闭'], fontproperties=CN_FONT)
fig.tight_layout(); fig.savefig(FIG/'lesson24_relu_gate_gradient.png',dpi=180,bbox_inches='tight'); plt.close(fig)

OUT.write_text('\n'.join(lines),encoding='utf-8')
