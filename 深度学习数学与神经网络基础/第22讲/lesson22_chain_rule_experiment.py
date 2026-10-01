import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).resolve().parent
FIG = OUT / 'figures'
FIG.mkdir(exist_ok=True)
np.set_printoptions(precision=6, suppress=True)

print('第22讲：Chain Rule 实验')
print('=' * 72)

# ------------------------------------------------------------
# 实验1：最简单的两段链 x -> y -> L
# y = 3x + 1, L = y^2
# ------------------------------------------------------------
print('\n[实验1] 两段链：x -> y -> L')
x = 2.0
y = 3.0 * x + 1.0
L = y ** 2

dL_dy = 2.0 * y
dy_dx = 3.0
dL_dx = dL_dy * dy_dx
print(f'x={x:.3f}, y={y:.3f}, L={L:.3f}')
print(f'dL/dy={dL_dy:.3f}, dy/dx={dy_dx:.3f}')
print(f'Chain Rule: dL/dx={dL_dx:.3f}')

# finite difference check
def f1(x):
    y = 3.0 * x + 1.0
    return y ** 2

eps = 1e-5
num = (f1(x + eps) - f1(x - eps)) / (2 * eps)
print(f'数值差分 dL/dx={num:.6f}, 误差={abs(num-dL_dx):.3e}')

# ------------------------------------------------------------
# 实验2：更长的链 x -> a -> b -> L
# a=2x, b=a+1, L=b^2
# ------------------------------------------------------------
print('\n[实验2] 更长的链：x -> a -> b -> L')
x2 = 2.0
a = 2.0 * x2
b = a + 1.0
L2 = b ** 2

dL_db = 2.0 * b
db_da = 1.0
da_dx = 2.0
dL_dx2 = dL_db * db_da * da_dx
print(f'x={x2:.3f} -> a={a:.3f} -> b={b:.3f} -> L={L2:.3f}')
print(f'局部导数: dL/db={dL_db:.3f}, db/da={db_da:.3f}, da/dx={da_dx:.3f}')
print(f'总导数: dL/dx={dL_dx2:.3f}')

# ------------------------------------------------------------
# 实验3：神经元链 w -> z -> a -> L
# z=wx+b, a=ReLU(z), L=(a-target)^2
# ------------------------------------------------------------
print('\n[实验3] 神经元链：w -> z -> ReLU -> L')
x3 = 2.0
w = 1.5
bias = -1.0
target = 4.0
z = w * x3 + bias
a3 = max(0.0, z)
L3 = (a3 - target) ** 2

dL_da = 2.0 * (a3 - target)
da_dz = 1.0 if z > 0 else 0.0
dz_dw = x3
dz_db = 1.0
dL_dw = dL_da * da_dz * dz_dw
dL_db = dL_da * da_dz * dz_db
print(f'x={x3:.3f}, w={w:.3f}, b={bias:.3f}, target={target:.3f}')
print(f'z={z:.3f}, a=ReLU(z)={a3:.3f}, L={L3:.3f}')
print(f'dL/da={dL_da:.3f}, da/dz={da_dz:.3f}, dz/dw={dz_dw:.3f}')
print(f'dL/dw={dL_dw:.3f}, dL/db={dL_db:.3f}')

# finite difference checks
def neuron_loss(w_value, b_value):
    z_value = w_value * x3 + b_value
    a_value = max(0.0, z_value)
    return (a_value - target) ** 2

num_w = (neuron_loss(w + eps, bias) - neuron_loss(w - eps, bias)) / (2 * eps)
num_b = (neuron_loss(w, bias + eps) - neuron_loss(w, bias - eps)) / (2 * eps)
print(f'数值差分 dL/dw={num_w:.6f}, dL/db={num_b:.6f}')

# ------------------------------------------------------------
# 实验4：ReLU 关闭时，链上的梯度被乘成 0
# ------------------------------------------------------------
print('\n[实验4] ReLU gate 关闭时的梯度')
w_neg = -1.0
b_neg = 0.0
z_neg = w_neg * x3 + b_neg
a_neg = max(0.0, z_neg)
L_neg = (a_neg - target) ** 2
dL_da_neg = 2.0 * (a_neg - target)
da_dz_neg = 1.0 if z_neg > 0 else 0.0
dL_dw_neg = dL_da_neg * da_dz_neg * x3
print(f'z={z_neg:.3f}, ReLU(z)={a_neg:.3f}, L={L_neg:.3f}')
print(f'dL/da={dL_da_neg:.3f}, da/dz={da_dz_neg:.3f}, 所以 dL/dw={dL_dw_neg:.3f}')

# ------------------------------------------------------------
# 实验5：分支图，多条路径的梯度要相加
# u=x^2, v=3x, L=u+v
# ------------------------------------------------------------
print('\n[实验5] 分支图：同一个 x 通过两条路径影响 L')
x5 = 2.0
u = x5 ** 2
v = 3.0 * x5
L5 = u + v

dL_du = 1.0
du_dx = 2.0 * x5
dL_dv = 1.0
dv_dx = 3.0
path1 = dL_du * du_dx
path2 = dL_dv * dv_dx
dL_dx5 = path1 + path2
print(f'u=x^2={u:.3f}, v=3x={v:.3f}, L=u+v={L5:.3f}')
print(f'路径1贡献={path1:.3f}, 路径2贡献={path2:.3f}')
print(f'dL/dx=路径1+路径2={dL_dx5:.3f}')

def f5(x):
    return x**2 + 3.0*x
num5 = (f5(x5 + eps) - f5(x5 - eps)) / (2 * eps)
print(f'数值差分 dL/dx={num5:.6f}')

# ------------------------------------------------------------
# 实验6：局部导数的连乘会随深度缩小或放大（预览）
# ------------------------------------------------------------
print('\n[实验6] 深链中局部导数连乘的尺度（预览）')
depths = np.arange(1, 21)
small = 0.5 ** depths
large = 1.5 ** depths
print(f'0.5^10={0.5**10:.8f}, 0.5^20={0.5**20:.10f}')
print(f'1.5^10={1.5**10:.6f}, 1.5^20={1.5**20:.6f}')
print('这只是 vanishing/exploding gradient 的最小直觉预览，后面会专门学习。')

# ---------------- figures ----------------
# Figure 1: simple chain sensitivity
xs = np.linspace(-2.0, 4.0, 300)
Ls = (3.0 * xs + 1.0) ** 2
plt.figure(figsize=(7.2, 4.5))
plt.plot(xs, Ls)
plt.scatter([x], [L], s=50)
plt.xlabel('x')
plt.ylabel('L(x)')
plt.title('Chain Rule example: L=(3x+1)^2')
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(FIG / 'lesson22_chain_loss_curve.png', dpi=180)
plt.close()

# Figure 2: ReLU gate and gradient
zs = np.linspace(-4, 4, 400)
relu = np.maximum(0, zs)
relu_grad = (zs > 0).astype(float)
plt.figure(figsize=(7.2, 4.5))
plt.plot(zs, relu, label='ReLU(z)')
plt.plot(zs, relu_grad, label="local derivative")
plt.xlabel('z')
plt.ylabel('value')
plt.title('ReLU as a local gradient gate')
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(FIG / 'lesson22_relu_gradient_gate.png', dpi=180)
plt.close()

# Figure 3: products through depth
plt.figure(figsize=(7.2, 4.5))
plt.semilogy(depths, small, marker='o', label='0.5^depth')
plt.semilogy(depths, large, marker='o', label='1.5^depth')
plt.xlabel('number of chained local derivatives')
plt.ylabel('absolute product (log scale)')
plt.title('Products of local derivatives across depth')
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(FIG / 'lesson22_gradient_product_depth.png', dpi=180)
plt.close()

print('\n图已保存：')
for p in sorted(FIG.glob('lesson22_*.png')):
    print(' -', p.name)
