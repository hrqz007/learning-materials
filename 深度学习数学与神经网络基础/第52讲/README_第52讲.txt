第52讲《Embedding 的深度学习本质》学习顺序

建议顺序：
1. 第52讲_讲义_Embedding的深度学习本质.pdf
2. 运行 lesson52_embedding_experiment.py
3. 对照 第52讲_实验实际输出.txt
4. 阅读 第52讲_实验设计_Embedding查表Shape梯度与位置.pdf
5. 完成 第52讲_练习题与答案.pdf
6. 最后用 第52讲_核心概念速查卡.pdf 复习

本讲最低掌握目标：
- 能解释 nn.Embedding(V,D) 的输入、输出、weight.shape 与参数量。
- 能手算 lookup，并解释 one-hot @ E 等价。
- 能判断 (B,T)->(B,T,D)。
- 能解释为什么只更新被访问的词表行、重复 Token 为什么累加梯度。
- 能区分 Token Embedding、位置机制和 Contextual Hidden State。

实验环境：CPU 即可。
