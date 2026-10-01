《面向 LLM / Agent Systems 科研的深度学习零基础课》
第06讲：Batch 与批量矩阵运算——从 (B,D) 走向 LLM 的 (B,T,D)

建议学习顺序：
1. 第06讲_讲义_Batch与批量矩阵运算.pdf
2. 第06讲_核心概念速查卡.pdf
3. 第06讲_实验设计_Batch三维Tensor与共享Linear.pdf
4. 运行 lesson06_batch_tensor_experiment.py
5. 对照 第06讲_实验实际输出.txt
6. 完成 第06讲_练习题与答案.pdf（先做题，再看答案）

运行代码：
python lesson06_batch_tensor_experiment.py

最低依赖：
- Python 3.x
- NumPy
- PyTorch（实验最后一部分使用；若未安装，前面的 NumPy 实验仍可运行）

本讲最低验收：
- 能解释 (B,D) 和 (B,T,D) 每个轴；
- 能判断 (B,T,D) @ (D,O) -> (B,T,O)；
- 能解释“所有样本/Token共享同一套 Weight/Bias”；
- 能解释为什么 Linear 本身不负责 Token 之间的信息交互；
- 能区分参数量与 Batch/Sequence 带来的计算量、激活显存；
- 能看懂 PyTorch nn.Linear 对三维 Tensor 的行为。

下一讲：第07讲 神经元究竟是什么。
