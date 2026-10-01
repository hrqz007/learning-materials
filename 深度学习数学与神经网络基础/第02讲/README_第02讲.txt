《面向 LLM / Agent Systems 科研的深度学习零基础课》
第 02 讲：Shape、索引、切片、转置与维度变化

建议学习顺序：

1. 第02讲_讲义_Shape索引切片转置与维度变化.pdf
   - 先理解 Index、Slice、Reshape、Transpose、expand_dims、squeeze。
   - 特别注意每一步操作前后的 Shape 与轴语义。

2. 第02讲_实验设计_Tensor索引切片Reshape与Transpose.pdf
   - 不要直接运行代码。
   - 先在纸上预测各表达式的输出 Shape。

3. lesson02_shape_index_slice_experiment.py
   - 完整可运行 NumPy 实验代码。
   - 每一部分都有中文注释。
   - 运行命令：
       python lesson02_shape_index_slice_experiment.py

4. 第02讲_实验实际输出.txt
   - 本材料制作时实际执行脚本得到的输出。
   - 用于检查自己的运行结果是否一致。

5. 第02讲_练习题与答案.pdf
   - 建议先独立完成练习题，再阅读答案。
   - 至少达到 80% 能独立解释后再继续下一讲。

6. 第02讲_核心概念速查卡.pdf
   - 后续阅读代码、做实验时随时查阅。

本讲最低验收标准：

A. 能解释为什么 X[0] 与 X[0:1] 的 Shape 不同；
B. 能解释为什么 X[:,1,:] 与 X[:,1:2,:] 的 Shape 不同；
C. 能说明 reshape 必须保持元素总数不变；
D. 能区分 reshape 与 transpose；
E. 能解释 np.expand_dims / squeeze 在做什么；
F. 面对 [B,T,D] Tensor，能够预测常见索引操作后的 Shape。

与 LLM / Transformer 的连接：
- logits[:, -1, :]：所有 batch 的最后一个 Token；
- Multi-Head Attention 中常通过 reshape + transpose 重新排列 Head / Token 轴；
- 后面学习 Q/K/V、Attention 时会反复使用本讲的 Shape 推理能力。

下一讲：
第 03 讲《向量、Dot Product 与“相似度”》
将第一次进入后续 Attention 的核心数学：两个向量为什么能通过点积得到一个匹配分数。
