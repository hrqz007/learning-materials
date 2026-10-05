《面向 LLM / Agent Systems 科研的深度学习零基础课》
第50讲：LSTM 与 GRU

建议学习顺序：
1. 第50讲_讲义_LSTM与GRU.pdf
   先完整学习主讲义。重点理解：Cell State、Forget/Input/Output Gate、GRU 的 Reset/Update Gate、长期 Gradient Flow、Shape。

2. 第50讲_核心概念速查卡.pdf
   学完讲义后用于复习，不建议替代主讲义。

3. 第50讲_实验设计_LSTM与GRU门控梯度与状态传递.pdf
   先看研究问题、控制变量和预期，再运行代码。

4. lesson50_lstm_gru_experiment.py
   直接运行：
   python lesson50_lstm_gru_experiment.py
   代码包含逐步中文注释，并会生成/复现 figures 中的实验图。

5. 第50讲_实验实际输出.txt
   用于核对自己的运行结果。不同 PyTorch / NumPy 版本可能存在极小浮点差异。

6. 第50讲_练习题与答案.pdf
   建议先遮住答案独立完成，再核对。

本讲最低验收标准：
- 能解释 LSTM 为什么同时有 h_t 和 c_t；
- 能解释 Forget / Input / Output Gate 的职责；
- 能说明为什么直接 Cell-State 路径有助于缓解长期梯度衰减，但不能保证无限记忆；
- 能解释 GRU 与 LSTM 的结构区别；
- 能判断 nn.LSTM / nn.GRU 的 (B,T,D) 输入输出 Shape；
- 能解释为什么 LSTM / GRU 仍然存在时间轴串行依赖；
- 能说明分块状态传递与 Truncated BPTT 的区别。

下一讲：第51讲《为什么 Transformer 最终压过 RNN：长期依赖、路径长度与并行计算》
