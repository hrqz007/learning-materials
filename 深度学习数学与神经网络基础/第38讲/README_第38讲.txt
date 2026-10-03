《面向 LLM / Agent Systems 科研的深度学习零基础课》
第38讲：Adam Optimizer

建议学习顺序：
1. 第38讲_讲义_Adam.pdf
2. 第38讲_核心概念速查卡.pdf
3. 第38讲_实验设计_Adam.pdf
4. 运行 lesson38_adam_experiment.py
5. 对照 第38讲_实验实际输出.txt 与 figures/
6. 完成 第38讲_练习题与答案.pdf

本讲最低验收：
- 能说清 m_t、v_t、m_hat、v_hat 分别是什么；
- 能解释 bias correction；
- 能解释 Adam 的坐标级 adaptive scaling；
- 能解释 betas、epsilon、learning rate 的角色；
- 能看懂 torch.optim.Adam 的关键参数与 optimizer state；
- 知道 zero_grad 不会清 Adam state；
- 知道 checkpoint 需保存 optimizer.state_dict() 才能更完整续训。
