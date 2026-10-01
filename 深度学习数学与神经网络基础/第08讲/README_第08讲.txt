《面向 LLM / Agent Systems 科研的深度学习零基础课》
第08讲：为什么仅有 Linear 不够

推荐学习顺序：
1. 第08讲_讲义_为什么仅有Linear不够.pdf
2. 第08讲_核心概念速查卡.pdf
3. 第08讲_实验设计_多层Linear折叠与非线性.pdf
4. 运行 lesson08_why_nonlinearity_experiment.py
5. 对照 第08讲_实验实际输出.txt 和 figures/ 下两张结果图
6. 独立完成 第08讲_练习题与答案.pdf

运行命令：
python lesson08_why_nonlinearity_experiment.py

最低依赖：
- Python 3.x
- NumPy
- Matplotlib

可选：
- PyTorch（最后一个对照实验）

本讲最低验收标准：
- 能手算两个 affine 层如何合并；
- 能写出 W_eq = W1 @ W2；
- 能写出 b_eq = b1 @ W2 + b2；
- 能解释 Bias 为什么不是非线性；
- 能解释为什么 ReLU 会破坏简单的 affine 折叠；
- 能把结论连接到 Transformer FFN 的 Linear -> Activation -> Linear。
