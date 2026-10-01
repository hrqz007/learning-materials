《面向 LLM / Agent Systems 科研的深度学习零基础课》
第17讲：Softmax - 怎样把 Logits 真正变成概率分布

建议学习顺序：
1. 第17讲_讲义_Softmax.pdf
2. lesson17_softmax_experiment.py（先阅读“预测”，再运行）
3. 第17讲_实验实际输出.txt（与自己的运行结果核对）
4. 第17讲_实验设计_SoftmaxTemperature与数值稳定性.pdf
5. 第17讲_练习题与答案.pdf（先做题，再看答案）
6. 第17讲_核心概念速查卡.pdf（最后复习）

本讲最低验收标准：
- 能从 3 个 logits 手算 Softmax。
- 能解释 Softmax 为什么使用 exp 和归一化。
- 能解释为什么减去 max 不改变结果却能提高数值稳定性。
- 能根据 Tensor Shape 判断 softmax 的 dim / axis。
- 能解释 Temperature 对分布尖锐程度的影响。
- 能区分 Softmax probability 与经过校准的真实置信度。
- 能把 Softmax 连接到 LLM vocabulary 输出层和 Attention weights。

运行实验：
    python lesson17_softmax_experiment.py

依赖：
    numpy
    matplotlib
    torch（可选；若没有 PyTorch，脚本会跳过对照实验）

实验生成三张图：
- figures/lesson17_logits_to_probabilities.png
- figures/lesson17_temperature_effect.png
- figures/lesson17_logit_gap_probability.png

下一讲：
第18讲 Cross Entropy - 从“正确答案概率”到分类 / Next-token Prediction Loss。
