第18讲：Cross Entropy - 从正确答案概率到 Next-token Prediction Loss

推荐学习顺序：
1. 第18讲_讲义_CrossEntropy.pdf
2. 先手算讲义里的 [2.0, 1.0, 0.1] 示例
3. 阅读 第18讲_实验设计_CrossEntropy与NextTokenLoss.pdf
4. 运行 lesson18_cross_entropy_experiment.py
5. 对照 第18讲_实验实际输出.txt
6. 完成 第18讲_练习题与答案.pdf
7. 最后用 第18讲_核心概念速查卡.pdf 复盘

本讲最低验收：
- 能解释 CE = -log(p_correct)
- 能解释 one-hot 公式为什么化简
- 能读懂 (B,C) 和 (B,T,V) 的类别维
- 知道 PyTorch cross_entropy 直接吃 logits
- 能说明 causal LM 的 next-token target shift
- 知道 lower CE 不等于 Agent 系统整体更可靠

运行代码：
  python lesson18_cross_entropy_experiment.py

代码依赖：NumPy、Matplotlib；PyTorch 仅用于结果对照。
