《面向 LLM / Agent Systems 科研的深度学习零基础课》
第23讲：Computational Graph 与局部梯度

建议学习顺序：
1. 第23讲_讲义_ComputationalGraph与局部梯度.pdf
2. 第23讲_核心概念速查卡.pdf
3. 第23讲_练习题与答案.pdf（先做题，再看答案）
4. 第23讲_实验设计_ComputationalGraph与局部梯度.pdf
5. 运行 lesson23_computational_graph_experiment.py
6. 对照 第23讲_实验实际输出.txt 与 figures/ 图像

本讲最低验收标准：
- 能说出 backward 统一模板：收到梯度 × 局部导数；
- 能写出 Add / Multiply / Square / ReLU 的最小 backward 规则；
- 能手工完成 x,w -> multiply -> +b -> ReLU -> loss 的一次完整 backward；
- 知道共享变量的梯度必须累加；
- 知道“有梯度”和“会被 optimizer 更新”不是一回事。

下一讲：第24讲 一个神经元的 Backpropagation。
