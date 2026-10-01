《面向 LLM / Agent Systems 科研的深度学习零基础课》
第05讲：Linear Transformation 与 y=Wx+b：Weight、Bias 到底是什么

推荐学习顺序：
1. 第05讲_讲义_LinearTransformation_Weight与Bias.pdf
2. 在纸上手算讲义中的 3->2 例子
3. 第05讲_实验设计_LinearWeightBias与PyTorchLinear.pdf
4. 运行 lesson05_linear_weight_bias_experiment.py
5. 对照 第05讲_实验实际输出.txt
6. 第05讲_练习题与答案.pdf（先做题再看答案）
7. 最后用 第05讲_核心概念速查卡.pdf 复习

本讲最低目标：
- 能解释 x、W、b、y 的含义；
- 能判断 Linear Layer 的输入/输出 Shape；
- 能手算 x @ W + b；
- 能计算参数数量；
- 能解释 Batch 共享参数与 Bias broadcasting；
- 能理解 PyTorch nn.Linear 的 weight 存储为 (out_features, in_features)；
- 能把本讲连接到 Transformer 的 Q/K/V projection。

运行环境：Python 3.x + NumPy；PyTorch 仅用于最后对照验证。
运行命令：python lesson05_linear_weight_bias_experiment.py
