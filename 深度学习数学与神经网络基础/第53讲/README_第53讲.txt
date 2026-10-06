第53讲《Q、K、V 是怎样算出来的》学习顺序

建议顺序：
1. 先读《第53讲_讲义_QKV是怎样算出来的.pdf》
2. 不看答案完成《第53讲_练习题与答案.pdf》的前两页题目
3. 阅读《第53讲_实验设计_QKVProjection.pdf》
4. 运行 lesson53_qkv_projection_experiment.py
5. 对照《第53讲_实验实际输出.txt》
6. 最后用《第53讲_核心概念速查卡.pdf》复习

最低验收标准：
- 能写出 Q=X@W_Q、K=X@W_K、V=X@W_V 的 Shape。
- 知道 Q/K/V 通常来自同一个 X，但使用三套独立可训练参数。
- 能解释为什么 QKV Projection 本身不发生 Token mixing。
- 能解释 PyTorch nn.Linear 的 weight Shape 为什么是 (out,in)。
- 知道三套 QKV Linear 可以融合成一次 Linear(D,3D)。
- 能预判 Q@K^T 的 Shape 是 (T,T)（忽略 Batch/Head 时）。

下一讲：
第54讲《Scaled Dot-Product Attention》
QK^T -> scale -> mask -> softmax -> weighted V
