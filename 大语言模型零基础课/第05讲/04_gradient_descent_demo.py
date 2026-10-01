# -*- coding: utf-8 -*-
"""
第5讲配套实验：梯度下降 + 数值梯度验证
不依赖第三方库，直接使用 Python 3 即可运行。

目标：训练极简模型 y_hat = w * x，让 w 学到接近 3。
数据：x = 2, y = 6，因此理想 w = 3。
"""

x = 2.0
y = 6.0
w = 0.5
learning_rate = 0.1


def loss_fn(w_value):
    y_hat = w_value * x
    return (y_hat - y) ** 2


def analytic_gradient(w_value):
    """L=(w*x-y)^2，所以 dL/dw = 2*(w*x-y)*x"""
    y_hat = w_value * x
    return 2.0 * (y_hat - y) * x


def numerical_gradient(w_value, eps=1e-5):
    """有限差分近似梯度，仅用于理解/检查，不是大模型训练的实际做法。"""
    return (loss_fn(w_value + eps) - loss_fn(w_value - eps)) / (2.0 * eps)


print("=== 先验证解析梯度与数值梯度 ===")
print(f"当前 w = {w:.6f}")
print(f"解析梯度 = {analytic_gradient(w):.6f}")
print(f"数值梯度 = {numerical_gradient(w):.6f}")
print()

print("=== 开始梯度下降 ===")
print("step |      w      | prediction |    loss    |  gradient")
print("-----+-------------+------------+------------+-----------")

for step in range(12):
    prediction = w * x
    loss = (prediction - y) ** 2
    grad = 2.0 * (prediction - y) * x

    print(f"{step:>4} | {w:>11.6f} | {prediction:>10.6f} | {loss:>10.6f} | {grad:>9.6f}")

    # 梯度下降更新：w_new = w_old - learning_rate * gradient
    w = w - learning_rate * grad

print("\n训练结束")
print(f"最终 w ≈ {w:.6f}")
print(f"理想 w = {y/x:.6f}")
print(f"最终预测 = {w*x:.6f}")
print(f"最终 loss = {loss_fn(w):.8f}")

print("\n观察重点：")
print("1. loss 是否总体越来越小？")
print("2. 当梯度为负时，减去负梯度会让 w 增大。")
print("3. w 会逐渐靠近 3，而不是程序员直接把答案 3 写进去。")
