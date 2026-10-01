《面向 LLM / Agent Systems 科研的深度学习零基础课》
第09讲：Activation Function：ReLU、Sigmoid、Tanh、GELU 与 SiLU

建议学习顺序：
1. 阅读《第09讲_讲义_ActivationFunctions.pdf》
2. 阅读《第09讲_核心概念速查卡.pdf》
3. 先不看答案，完成《第09讲_练习题与答案.pdf》前半部分
4. 阅读《第09讲_实验设计_ActivationFunctions.pdf》中的实验问题与预测
5. 运行 lesson09_activation_functions_experiment.py
6. 将自己的预测与《第09讲_实验实际输出.txt》比较
7. 查看 figures/ 下的三张图
8. 最后回看练习答案并复述本讲核心链条

运行代码：
    python lesson09_activation_functions_experiment.py

依赖：
    numpy
    matplotlib
    torch（可选；只用于最后 PyTorch 对照）

最低验收标准：
- 能解释 Activation 为什么引入非线性；
- 能手算 ReLU；
- 理解 Sigmoid/Tanh 的输出范围与饱和直觉；
- 理解 GELU/SiLU 与 ReLU 对负值处理的差异；
- 知道常见逐元素 Activation 通常不改变 Tensor Shape；
- 能把本讲连接到 Transformer FFN：Linear -> Activation -> Linear。
