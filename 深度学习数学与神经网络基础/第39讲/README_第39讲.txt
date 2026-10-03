《面向 LLM / Agent Systems 科研的深度学习零基础课》
第39讲：AdamW 与 Weight Decay

推荐学习顺序：
1. 第39讲_讲义_AdamW与WeightDecay.pdf
2. 第39讲_核心概念速查卡.pdf
3. 第39讲_练习题与答案.pdf（建议先做题后看答案）
4. 第39讲_实验设计_AdamW与WeightDecay.pdf
5. lesson39_adamw_weight_decay_experiment.py
6. 对照 第39讲_实验实际输出.txt 与 figures/ 下的结果图

本讲最低验收：
- 能解释 L2 regularization 为什么给 Gradient 增加 lambda * theta；
- 能证明基础 SGD 下 L2 与 multiplicative weight decay 的一步更新等价；
- 能解释为什么 Adam 中 coupled L2 会进入 m/v state；
- 能写出 AdamW 的 decoupled weight decay 直觉公式；
- 知道 weight_decay 的实际作用还受 learning rate、scheduler、optimizer steps、parameter groups 影响；
- 能看懂 PyTorch AdamW 的基本配置并判断哪些参数是否被 decay。

运行代码：
python lesson39_adamw_weight_decay_experiment.py

依赖：
Python 3.x
NumPy
PyTorch
Matplotlib

说明：
本材料中的玩具回归实验用于验证优化机制，不用于证明 AdamW 在所有任务上都比 Adam 泛化更好。
