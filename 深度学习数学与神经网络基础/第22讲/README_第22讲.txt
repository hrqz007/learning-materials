《面向 LLM / Agent Systems 科研的深度学习零基础课》
第22讲：Chain Rule（链式法则）

建议学习顺序：
1. 第22讲_讲义_ChainRule链式法则.pdf
2. 第22讲_核心概念速查卡.pdf
3. 第22讲_练习题与答案.pdf（先做题，再看答案）
4. 第22讲_实验设计_ChainRule与梯度路径.pdf
5. 运行 lesson22_chain_rule_experiment.py
6. 对照 第22讲_实验实际输出.txt 与 figures/ 图像

本讲最低验收标准：
- 能解释为什么单条路径上的局部导数相乘；
- 能解释为什么分支图的梯度贡献相加；
- 能手算 x->y->L 的 Chain Rule；
- 能手算 w->z->ReLU->Loss 的 dL/dw；
- 知道 Backpropagation 是在计算图上系统应用 Chain Rule。

下一讲：第23讲 Computational Graph 与局部梯度。
