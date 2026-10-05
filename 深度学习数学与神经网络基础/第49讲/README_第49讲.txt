《面向 LLM / Agent Systems 科研的深度学习零基础课》
第49讲：Sequence 与 RNN

建议学习顺序：
1. 第49讲_讲义_Sequence与RNN.pdf
2. lesson49_rnn_experiment.py（亲自运行）
3. 第49讲_实验实际输出.txt（对照）
4. 第49讲_实验设计_HiddenState参数共享与BPTT.pdf
5. 第49讲_练习题与答案.pdf
6. 第49讲_核心概念速查卡.pdf

本讲最低验收标准：
- 能写出 h_t = tanh(W_ih x_t + W_hh h_{t-1} + b)；
- 能解释 Hidden State 是历史的递推压缩状态，不是完整历史；
- 能判断 nn.RNN(batch_first=True) 的 (B,T,D)->(B,T,H) Shape；
- 能解释参数为什么跨时间共享，参数量为什么不随 T 增长；
- 能说明 BPTT 与普通 Backpropagation 的关系；
- 能用 a^T 的连续乘积直觉解释长期梯度消失/爆炸；
- 能解释 RNN 时间维为什么难以完全并行。
