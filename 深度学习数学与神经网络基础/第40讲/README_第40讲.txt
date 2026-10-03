《面向 LLM / Agent Systems 科研的深度学习零基础课》
第40讲：Initialization、Vanishing Gradient 与 Exploding Gradient

建议学习顺序：
1. 第40讲_讲义_Initialization与GradientStability.pdf
2. 第40讲_核心概念速查卡.pdf
3. 第40讲_实验设计_Initialization与GradientStability.pdf
4. lesson40_initialization_gradients_experiment.py
5. 第40讲_实验实际输出.txt
6. 第40讲_练习题与答案.pdf

运行代码：
python lesson40_initialization_gradients_experiment.py

依赖：
- numpy
- torch
- matplotlib

本讲最低闭环：
- 能解释为什么隐藏层 Weight 不能全部为 0；
- 能解释随机初始化为什么还要控制尺度；
- 能说清 Xavier / He 的适用直觉；
- 能从 Chain Rule 解释梯度消失和爆炸；
- 知道 Gradient Clipping 只能限制异常更新，不等于解决根因；
- 能看懂 PyTorch 初始化和 clip_grad_norm_ 代码。
