第11讲：Forward Propagation 与 Computational Graph

建议学习顺序：
1. 第11讲_讲义_ForwardPropagation与ComputationalGraph.pdf
2. 运行 lesson11_forward_computational_graph_experiment.py
3. 对照 第11讲_实验实际输出.txt
4. 阅读 第11讲_实验设计_Forward与计算图.pdf
5. 用 第11讲_核心概念速查卡.pdf 复习
6. 最后完成 第11讲_练习题与答案.pdf

运行命令：
python lesson11_forward_computational_graph_experiment.py

依赖：
- Python 3.x
- NumPy
- PyTorch（可选；未安装时只跳过最后一个对照实验）

本讲最低验收：
- 能解释 Forward Propagation 是什么；
- 能区分结构图和 Computational Graph；
- 能把一个 MLP 写成 x -> z1 -> a1 -> y；
- 能同时追踪每一步数值、Shape 和依赖；
- 知道 Backpropagation 将沿这张依赖图反方向计算梯度。
