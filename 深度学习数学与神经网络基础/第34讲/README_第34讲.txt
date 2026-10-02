《面向 LLM / Agent Systems 科研的深度学习零基础课》
第34讲：SGD 与 Mini-Batch SGD

建议学习顺序：
1. 第34讲_讲义_SGD与MiniBatchSGD.pdf
2. lesson34_sgd_minibatch_experiment.py（建议先预测结果，再运行）
3. 第34讲_实验实际输出.txt（与自己的运行结果对照）
4. 第34讲_实验设计_SGD与MiniBatch.pdf（从科研实验角度复盘变量、假设与边界）
5. 第34讲_练习题与答案.pdf
6. 第34讲_核心概念速查卡.pdf（最后用于复习）

本讲最低闭环：
- 能区分 Full-Batch GD、严格 SGD、Mini-Batch SGD；
- 能解释 Mini-Batch Gradient 为什么有噪声；
- 会计算 steps/epoch = ceil(N / B)；
- 知道 DataLoader 决定 batch，Optimizer 决定参数更新规则；
- 能解释 Gradient Accumulation 与 Effective Batch；
- 比较不同 Batch Size 时，会主动检查 LR、Step、Epoch、seen samples/tokens 与计算预算是否公平。

运行环境：
- Python 3
- NumPy
- PyTorch
- Matplotlib

运行命令示例：
python lesson34_sgd_minibatch_experiment.py

脚本会在 figures/ 下生成实验图，并将关键数值写入“第34讲_实验实际输出.txt”。
