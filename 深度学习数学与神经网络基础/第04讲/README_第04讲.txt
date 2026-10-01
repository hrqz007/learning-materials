《面向 LLM / Agent Systems 科研的深度学习零基础课》
第04讲：Matrix Multiplication —— 为什么神经网络几乎到处都在做矩阵乘法

【建议学习顺序】
1. 第04讲_讲义_MatrixMultiplication矩阵乘法.pdf
2. 在纸上手算讲义中的 2×3 @ 3×2 示例
3. 第04讲_实验设计_矩阵乘法与QKV投影.pdf
4. 运行 lesson04_matrix_multiplication_experiment.py
5. 将自己的预测与 第04讲_实验实际输出.txt 对照
6. 完成 第04讲_练习题与答案.pdf（先做题，后看答案）
7. 最后使用 第04讲_核心概念速查卡.pdf 复习

【运行代码】
环境：Python 3.x + NumPy；PyTorch 部分为可选验证。
命令：
    python lesson04_matrix_multiplication_experiment.py

【本讲必须掌握】
1. (m,n) @ (n,p) -> (m,p)
2. 中间维必须相等，因为输出元素来自“行 dot 列”
3. C[i,j] = A 的第 i 行 dot B 的第 j 列
4. A * B（逐元素乘法）与 A @ B（矩阵乘法）不同
5. X @ W 可以一次处理一个 Batch 的全部样本
6. Q=XW_Q、K=XW_K、V=XW_V 本质上都是可学习矩阵投影

【与前后课程连接】
第03讲 Dot Product -> 第04讲 Matrix Multiplication -> 第05讲 y = Wx + b
后续 Attention 中 QK^T 也完全建立在本讲上。
