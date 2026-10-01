第19讲：Partial Derivative（偏导数）

推荐学习顺序：
1. 第19讲_讲义_PartialDerivative偏导数.pdf
2. 手算 f(x,y)=x^2+3xy+y^2 在 (2,1) 的两个偏导
3. 阅读 第19讲_实验设计_PartialDerivative与多参数Loss.pdf
4. 运行 lesson19_partial_derivative_experiment.py
5. 对照 第19讲_实验实际输出.txt
6. 完成 第19讲_练习题与答案.pdf
7. 最后用 第19讲_核心概念速查卡.pdf 复盘

最低验收：
- 能解释 ∂L/∂w 的自然语言含义
- 能说明求一个偏导时其他变量为什么暂时固定
- 能解释偏导正/负/零的局部意义
- 能用 ΔL≈(∂L/∂w)Δw 做小扰动预测
- 知道数值差分可用于 Gradient Check
- 知道下一讲会把所有偏导数组成 Gradient

运行代码：
  python lesson19_partial_derivative_experiment.py

代码依赖：NumPy、Matplotlib；不需要 GPU。
