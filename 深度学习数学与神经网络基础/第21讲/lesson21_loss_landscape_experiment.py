import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=6, suppress=True)

# ============================================================
# 第21讲实验：Loss Landscape
# 目标：观察低谷、鞍点、平坦区、陡峭区以及 gradient≈0 的不同含义
# ============================================================


def bowl_loss(w1, w2):
    """简单凸碗形 Loss：唯一最低点在 (1, -0.5)。"""
    return (w1 - 1.0) ** 2 + 2.0 * (w2 + 0.5) ** 2


def bowl_grad(w1, w2):
    return np.array([2.0 * (w1 - 1.0), 4.0 * (w2 + 0.5)])


def saddle_loss(x, y):
    """标准鞍点：原点 gradient=0，但原点不是局部最小值。"""
    return x ** 2 - y ** 2


def saddle_grad(x, y):
    return np.array([2.0 * x, -2.0 * y])


def flat_loss(w):
    """同样最低点在 0，但更平缓。"""
    return 0.2 * w ** 2


def sharp_loss(w):
    """同样最低点在 0，但更陡峭。"""
    return 5.0 * w ** 2


def plateau_loss(w):
    """远离 0 时 Loss 仍较高，但梯度会非常小。"""
    return np.tanh(w) ** 2


def plateau_grad(w):
    # d/dw tanh(w)^2 = 2*tanh(w)*(1-tanh(w)^2)
    t = np.tanh(w)
    return 2.0 * t * (1.0 - t ** 2)


def central_difference_2d(func, point, eps=1e-5):
    point = np.array(point, dtype=float)
    grad = np.zeros_like(point)
    for i in range(len(point)):
        p_plus = point.copy()
        p_minus = point.copy()
        p_plus[i] += eps
        p_minus[i] -= eps
        grad[i] = (func(*p_plus) - func(*p_minus)) / (2 * eps)
    return grad


print("=" * 70)
print("实验1：凸碗形 Loss Landscape 与唯一低谷")
print("=" * 70)
point = np.array([0.0, 1.0])
loss = bowl_loss(*point)
grad = bowl_grad(*point)
num_grad = central_difference_2d(bowl_loss, point)
print("当前点:", point)
print("Loss:", loss)
print("解析 Gradient:", grad)
print("数值 Gradient:", num_grad)
print("最大误差:", np.max(np.abs(grad - num_grad)))
print("最低点应位于: [1.0, -0.5]")
print()

print("=" * 70)
print("实验2：Gradient = 0 不等于局部最小值——鞍点")
print("=" * 70)
origin = np.array([0.0, 0.0])
print("原点 Loss:", saddle_loss(*origin))
print("原点 Gradient:", saddle_grad(*origin))
for p in [(0.2, 0.0), (0.0, 0.2), (0.5, 0.0), (0.0, 0.5)]:
    print(f"点 {p} 的 Loss = {saddle_loss(*p):.6f}")
print("解释：沿 x 方向离开原点 Loss 上升；沿 y 方向离开原点 Loss 下降。")
print("因此原点 gradient=0，但不是局部最小值。")
print()

print("=" * 70)
print("实验3：Flat 与 Sharp——同一个最低点，不同局部陡峭程度")
print("=" * 70)
for w in [0.1, 0.5, 1.0]:
    print(
        f"w={w:>3}: flat_loss={flat_loss(w):.6f}, "
        f"sharp_loss={sharp_loss(w):.6f}"
    )
print("两条曲线最低点都是 w=0，但 sharp 曲线离开最低点后上升得更快。")
print("注意：这个玩具实验只说明曲率/陡峭程度不同，不直接证明泛化优劣。")
print()

print("=" * 70)
print("实验4：高 Loss 平台也可能有很小 Gradient")
print("=" * 70)
for w in [0.0, 1.0, 3.0, 5.0]:
    print(
        f"w={w:>3}: Loss={plateau_loss(w):.8f}, "
        f"|gradient|={abs(plateau_grad(w)):.8f}"
    )
print("在 w=5 附近，Loss 接近 1，但 gradient 已非常小。")
print("所以 'gradient 很小' 不能单独推出 '已经找到好解'。")
print()

print("=" * 70)
print("实验5：从不同起点沿负梯度走向同一个低谷（简单凸例子）")
print("=" * 70)
starts = [np.array([-2.0, 2.0]), np.array([3.0, 2.0]), np.array([-1.0, -2.0])]
lr = 0.15
for idx, start in enumerate(starts, 1):
    w = start.copy()
    for _ in range(25):
        w = w - lr * bowl_grad(*w)
    print(
        f"起点{idx} {start} -> 25步后 {w}, "
        f"Loss={bowl_loss(*w):.8f}"
    )
print()

# ------------------------------------------------------------
# Figure 1: bowl landscape contour
# ------------------------------------------------------------
x = np.linspace(-2.5, 3.5, 250)
y = np.linspace(-2.5, 2.5, 250)
X, Y = np.meshgrid(x, y)
Z = bowl_loss(X, Y)
plt.figure(figsize=(7, 5.2))
cs = plt.contour(X, Y, Z, levels=18)
plt.clabel(cs, inline=True, fontsize=7)
plt.scatter([1.0], [-0.5], s=50, label='minimum (1, -0.5)')
plt.scatter([0.0], [1.0], s=45, marker='x', label='example point')
g = bowl_grad(0.0, 1.0)
plt.quiver([0.0], [1.0], [g[0]], [g[1]], angles='xy', scale_units='xy', scale=7, label='gradient direction')
plt.xlabel('w1')
plt.ylabel('w2')
plt.title('Convex bowl loss landscape')
plt.legend()
plt.tight_layout()
plt.savefig('/mnt/data/deep_learning_lesson21/final/figures/lesson21_bowl_contour.png', dpi=180)
plt.close()

# Figure 2: saddle contour
x = np.linspace(-2, 2, 260)
y = np.linspace(-2, 2, 260)
X, Y = np.meshgrid(x, y)
Z = saddle_loss(X, Y)
plt.figure(figsize=(7, 5.2))
cs = plt.contour(X, Y, Z, levels=np.linspace(-3, 3, 17))
plt.clabel(cs, inline=True, fontsize=7)
plt.scatter([0], [0], s=55, label='saddle: gradient = 0')
plt.axhline(0, linewidth=0.8)
plt.axvline(0, linewidth=0.8)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Saddle point: L(x,y)=x^2-y^2')
plt.legend()
plt.tight_layout()
plt.savefig('/mnt/data/deep_learning_lesson21/final/figures/lesson21_saddle_contour.png', dpi=180)
plt.close()

# Figure 3: flat vs sharp cross sections
w = np.linspace(-2, 2, 400)
plt.figure(figsize=(7, 5.2))
plt.plot(w, flat_loss(w), label='flat: 0.2 w^2')
plt.plot(w, sharp_loss(w), label='sharp: 5 w^2')
plt.xlabel('w')
plt.ylabel('Loss')
plt.title('Same minimum, different sharpness')
plt.legend()
plt.tight_layout()
plt.savefig('/mnt/data/deep_learning_lesson21/final/figures/lesson21_flat_vs_sharp.png', dpi=180)
plt.close()

# Figure 4: plateau
w = np.linspace(-6, 6, 500)
plt.figure(figsize=(7, 5.2))
plt.plot(w, plateau_loss(w), label='Loss = tanh(w)^2')
plt.plot(w, np.abs(plateau_grad(w)), label='|gradient|')
plt.xlabel('w')
plt.ylabel('Value')
plt.title('High-loss plateau can have tiny gradient')
plt.legend()
plt.tight_layout()
plt.savefig('/mnt/data/deep_learning_lesson21/final/figures/lesson21_plateau_small_gradient.png', dpi=180)
plt.close()

# Figure 5: multiple gradient-descent trajectories in bowl landscape
x = np.linspace(-2.5, 3.5, 250)
y = np.linspace(-2.5, 2.5, 250)
X, Y = np.meshgrid(x, y)
Z = bowl_loss(X, Y)
plt.figure(figsize=(7, 5.2))
plt.contour(X, Y, Z, levels=18)
for start in starts:
    traj = [start.copy()]
    cur = start.copy()
    for _ in range(18):
        cur = cur - lr * bowl_grad(*cur)
        traj.append(cur.copy())
    traj = np.array(traj)
    plt.plot(traj[:, 0], traj[:, 1], marker='o', markersize=2.5)
plt.scatter([1.0], [-0.5], s=55, label='minimum')
plt.xlabel('w1')
plt.ylabel('w2')
plt.title('Different starts descend toward the same basin')
plt.legend()
plt.tight_layout()
plt.savefig('/mnt/data/deep_learning_lesson21/final/figures/lesson21_descent_trajectories.png', dpi=180)
plt.close()
