# 第6讲小实验：用多项式拟合观察欠拟合、合理拟合和过拟合
# 依赖：numpy、matplotlib

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# 真实规律：y = sin(2πx)，再加入随机噪声
x_train = np.linspace(0, 1, 18)
y_train = np.sin(2 * np.pi * x_train) + rng.normal(0, 0.18, size=len(x_train))

x_test = np.linspace(0.02, 0.98, 120)
y_test = np.sin(2 * np.pi * x_test)

# 比较三种复杂度
# 1阶：通常欠拟合
# 5阶：通常比较合适
# 15阶：很容易把训练噪声也“记住”，出现过拟合
for degree in [1, 5, 15]:
    coef = np.polyfit(x_train, y_train, degree)
    model = np.poly1d(coef)

    train_pred = model(x_train)
    test_pred = model(x_test)

    train_mse = np.mean((train_pred - y_train) ** 2)
    test_mse = np.mean((test_pred - y_test) ** 2)

    print(f"degree={degree:2d} | train MSE={train_mse:.4f} | test MSE={test_mse:.4f}")

# 画图：注意不要只看训练点拟合得多紧，而要看整体曲线是否接近真实规律
fig = plt.figure(figsize=(9, 6))
plt.scatter(x_train, y_train, label="training samples")
plt.plot(x_test, y_test, linewidth=2, label="true function")

for degree in [1, 5, 15]:
    coef = np.polyfit(x_train, y_train, degree)
    model = np.poly1d(coef)
    plt.plot(x_test, model(x_test), label=f"degree={degree}")

plt.ylim(-2, 2)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Underfitting vs. Good Fit vs. Overfitting")
plt.legend()
plt.tight_layout()
plt.savefig("overfitting_demo.png", dpi=160)
print("\n已生成图：overfitting_demo.png")
