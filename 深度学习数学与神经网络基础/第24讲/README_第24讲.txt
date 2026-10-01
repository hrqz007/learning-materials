《面向 LLM / Agent Systems 科研的深度学习零基础课》
第24讲：一个神经元的 Backpropagation

【建议学习顺序】
1. 第24讲_讲义_一个神经元的Backpropagation.pdf
2. 第24讲_核心概念速查卡.pdf
3. 第24讲_练习题与答案.pdf（建议先做题，再看答案）
4. 第24讲_实验设计_单神经元Backpropagation.pdf
5. lesson24_single_neuron_backprop_experiment.py
6. 第24讲_实验实际输出.txt
7. figures/ 下的实验结果图

【运行实验】
环境：Python 3.x
依赖：numpy、matplotlib；PyTorch 部分为可选验证。
运行：
python lesson24_single_neuron_backprop_experiment.py

【本讲最低验收标准】
- 能从 z=x·w+b、a=ReLU(z)、L=(a-y)^2 手工算出 Forward。
- 能解释 dL/dw_i=(dL/dz)x_i 为什么成立。
- 能解释 dL/dx_i=(dL/dz)w_i 为什么必须计算。
- 知道 dL/db=dL/dz 的原因。
- 能区分 parameter gradient、input gradient、intermediate gradient。
- 知道 ReLU gate 可以让一条路径的梯度归零。
- 知道 Gradient Check 是正确性验证工具，不是训练方法。

【与后续课程的连接】
第25讲将把本讲的单神经元规则扩展到两层 MLP：
Linear -> ReLU -> Linear -> Loss
并开始出现真正的向量/矩阵形式 Backpropagation。
