第51讲：为什么 Transformer 最终压过 RNN

建议学习顺序：
1. 第51讲_讲义_为什么Transformer压过RNN.pdf
2. lesson51_transformer_vs_rnn_experiment.py（建议先预测结果，再运行）
3. 第51讲_实验实际输出.txt
4. 第51讲_实验设计_RNN与Transformer路径并行与T平方代价.pdf
5. 第51讲_练习题与答案.pdf
6. 第51讲_核心概念速查卡.pdf

运行：
python lesson51_transformer_vs_rnn_experiment.py

本讲最低验收：
- 能解释 RNN 时间路径为什么随距离增长；
- 能解释 Self-Attention 为什么允许直接长程交互；
- 能区分训练并行与自回归生成串行；
- 能从 (T,d)@(d,T) 推出 T^2 score；
- 不把总 FLOPs、串行深度、显存、延迟、吞吐混为一谈。
