《面向 LLM / Agent Systems 科研的深度学习零基础课》
第41讲：Overfitting 与 Generalization

建议学习顺序：
1. 第41讲_讲义_Overfitting与Generalization.pdf
2. 第41讲_核心概念速查卡.pdf
3. 第41讲_实验设计_Overfitting与Generalization.pdf
4. lesson41_overfitting_generalization_experiment.py
5. 第41讲_实验实际输出.txt
6. 第41讲_练习题与答案.pdf

运行代码：
python lesson41_overfitting_generalization_experiment.py

依赖：
- numpy
- torch
- matplotlib

本讲最低闭环：
- 能区分 Optimization Success 与 Generalization；
- 能根据 Train/Validation 曲线识别典型过拟合；
- 能解释 Generalization Gap；
- 知道 Validation 用于模型选择，Test 不应反复调参；
- 能解释 Random Labels 为什么证明“训练集记忆 != 泛化”；
- 能说出 Agent trajectory / benchmark contamination 等数据泄漏风险；
- 知道 Early Stopping 是模型选择策略，不是因果规律证明。
