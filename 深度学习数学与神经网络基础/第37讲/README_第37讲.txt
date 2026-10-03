《面向 LLM / Agent Systems 科研的深度学习零基础课》
第37讲：Momentum 动量

建议学习顺序：
1. 第37讲_讲义_Momentum.pdf
2. lesson37_momentum_experiment.py（先读代码，再运行）
3. 第37讲_实验实际输出.txt
4. 第37讲_实验设计_Momentum.pdf
5. 第37讲_练习题与答案.pdf
6. 第37讲_核心概念速查卡.pdf

本讲最低验收标准：
- 能解释 v_t = μ v_(t-1) + g_t；
- 能手算 3~5 步 Momentum；
- 能解释为什么持续方向会积累、交替方向会抵消；
- 能解释为什么 μ 越大不一定越好；
- 能读懂 torch.optim.SGD(..., momentum=0.9)；
- 能说明 Momentum 与 Adam first moment 的关系。
