《面向 LLM / Agent Systems 科研的深度学习零基础课》
第47讲：LayerNorm 与 RMSNorm

推荐学习顺序：
1. 第47讲_讲义_LayerNorm与RMSNorm.pdf
2. 运行 lesson47_layernorm_rmsnorm_experiment.py
3. 对照 第47讲_实验实际输出.txt 和 figures/ 中的实验图
4. 阅读 第47讲_实验设计_LayerNorm与RMSNorm.pdf
5. 完成 第47讲_练习题与答案.pdf
6. 最后用 第47讲_核心概念速查卡.pdf 复习

本讲最低掌握目标：
- 能解释 LayerNorm 与 RMSNorm 的公式差异；
- 能对 (B,T,D) 判断归一化统计轴；
- 能解释 LN/RMSNorm 为什么不依赖 Batch 同伴；
- 能解释 train/eval 与 BatchNorm 的差异；
- 能解释 Pre-Norm / Post-Norm 与 Norm 类型不是同一维度的概念。

运行环境：Python 3.x, PyTorch, NumPy, Matplotlib。
