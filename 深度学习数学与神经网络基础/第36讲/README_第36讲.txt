《面向 LLM / Agent Systems 科研的深度学习零基础课》
第36讲：Learning Rate 学习率

建议学习顺序：
1. 第36讲_讲义_LearningRate.pdf
2. 第36讲_核心概念速查卡.pdf
3. 第36讲_练习题与答案.pdf（先做题后看答案）
4. 第36讲_实验设计_LearningRate.pdf
5. lesson36_learning_rate_experiment.py
6. 第36讲_实验实际输出.txt
7. figures/ 中的实验图

运行方式：
python lesson36_learning_rate_experiment.py

依赖：
- Python 3.x
- NumPy
- Matplotlib
- PyTorch

本讲最低闭环：
- 能解释 theta <- theta - lr * grad；
- 能区分 Gradient、Learning Rate、Update；
- 知道 LR 太大为何会振荡/发散；
- 知道曲率会限制稳定步长；
- 知道 SGD 与 Adam 的 LR 数值不能机械比较；
- 看懂 LLM 配置中的 peak LR / warmup / cosine schedule。
