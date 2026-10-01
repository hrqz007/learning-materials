《面向 LLM / Agent Systems 科研的深度学习零基础课》
第03讲：向量、Dot Product 与“相似度”

建议学习顺序：
1. 第03讲_讲义_向量_DotProduct与相似度.pdf
2. 不看代码，手算 [1,2,3]·[4,5,6]
3. 第03讲_实验设计_DotProduct方向尺度与Attention打分.pdf
4. 运行 lesson03_dot_product_experiment.py
5. 与 第03讲_实验实际输出.txt 对照
6. 第03讲_练习题与答案.pdf（先做题，再看答案）
7. 最后用 第03讲_核心概念速查卡.pdf 复习

运行命令：
python lesson03_dot_product_experiment.py

最低依赖：Python 3.x + NumPy
PyTorch 是可选对照；如果未安装，脚本会跳过 PyTorch 部分，不影响主实验。

本讲验收：
- 会手算点积；
- 知道 (D,)·(D,) -> Scalar；
- 能解释点积同时受到“方向 + 长度”影响；
- 能区分 Dot Product 与 Cosine Similarity；
- 能解释 Q·K 为什么能产生 Attention 的基础匹配分数；
- 能看懂一个 Query 对多个 Key 的 score 向量。
