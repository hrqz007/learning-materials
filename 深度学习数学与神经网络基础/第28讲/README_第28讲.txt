《面向 LLM / Agent Systems 科研的深度学习零基础课》
第28讲：PyTorch 中的 Linear 与 Activation

建议学习顺序：
1. 第28讲_讲义_PyTorchLinear与Activation.pdf
2. lesson28_linear_activation_experiment.py（先自己预测输出和Shape，再运行）
3. 第28讲_实验实际输出.txt（与自己的预测对照）
4. 第28讲_实验设计_PyTorchLinear与Activation.pdf
5. 第28讲_练习题与答案.pdf（先做题，后看答案）
6. 第28讲_核心概念速查卡.pdf（复习时使用）

本讲最低验收标准：
- 能解释 nn.Linear(Din, Dout) 的 weight.shape 为什么是 (Dout, Din)。
- 能把 linear(X) 还原为 X @ weight.T + bias。
- 能计算 Linear Layer 的参数量。
- 能预测二维 (B,D) 与三维 (B,T,D) 经过 Linear 后的Shape。
- 知道 ReLU / GELU / SiLU 通常改变数值而不改变Shape。
- 能解释 Transformer FFN 中 Linear -> Activation -> Linear 的基本作用。

运行代码：
python lesson28_linear_activation_experiment.py

依赖：
- Python 3
- NumPy
- PyTorch
- Matplotlib

说明：
- 本讲实验只验证 Forward、Shape、参数封装与数值一致性。
- 正式的 Autograd、loss.backward()、梯度累积等内容放在第29讲。
