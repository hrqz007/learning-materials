《面向 LLM / Agent Systems 科研的深度学习零基础课》
第43讲：Regularization 与 Weight Decay

建议学习顺序：
1. 第43讲_讲义_Regularization与WeightDecay.pdf
2. 第43讲_核心概念速查卡.pdf
3. 第43讲_练习题与答案.pdf（先独立做，再看答案）
4. 第43讲_实验设计_Regularization与WeightDecay.pdf
5. lesson43_regularization_weight_decay_experiment.py
6. 第43讲_实验实际输出.txt
7. figures/ 中的实验图

本讲最低闭环：
- 能解释 Regularization 为什么可能让 Train Loss 变差但 Test 变好；
- 能读懂 J = L_data + lambda/2 * ||theta||^2；
- 能解释 Weight Decay 太弱 / 适度 / 太强的典型现象；
- 能区分 L2、SGD weight decay 与 AdamW decoupled weight decay；
- 能用 Validation 选择正则强度，并保持 Test 独立；
- 能设计一个最基本的 Weight Decay Ablation。
