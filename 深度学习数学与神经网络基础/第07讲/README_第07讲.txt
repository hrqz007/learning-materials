《面向 LLM / Agent Systems 科研的深度学习零基础课》
第07讲：神经元究竟是什么：Weight、Bias 与加权求和

建议学习顺序：
1. 第07讲_讲义_神经元WeightBias与加权求和.pdf
2. 第07讲_核心概念速查卡.pdf
3. 第07讲_实验设计_拆开一个人工神经元.pdf
4. 运行 lesson07_neuron_experiment.py
5. 对照 第07讲_实验实际输出.txt
6. 完成 第07讲_练习题与答案.pdf（先做题，再看答案）

运行命令：
python lesson07_neuron_experiment.py

最低依赖：
- Python 3.x
- numpy

可选依赖：
- torch（只用于最后一个 PyTorch 对照实验；没有也不影响 NumPy 主实验）

本讲最低验收：
- 能手算 z = x·w + b；
- 能解释每个 x_i*w_i 的含义；
- 能解释 Weight 正负、Bias、参数共享；
- 能把多个神经元写成 x@W+b；
- 知道 Weight 大小不能直接等同于现实因果重要性；
- 知道 z 目前只是 pre-activation，下一步还要学习为什么需要非线性。
