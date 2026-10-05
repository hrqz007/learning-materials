《面向 LLM / Agent Systems 科研的深度学习零基础课》
第48讲：CNN 基础——Kernel、Convolution、Feature Map 与参数共享

建议学习顺序：
1. 先读《第48讲_讲义_CNN基础.pdf》，重点看图像 Tensor、Kernel 手算、Shape 公式、多 Channel、参数共享。
2. 再读《第48讲_核心概念速查卡.pdf》做第一次复习。
3. 打开《第48讲_实验设计_CNN基础.pdf》，先预测每个实验结果，不要直接看输出。
4. 运行 lesson48_cnn_basics_experiment.py。
5. 将自己的输出与《第48讲_实验实际输出.txt》逐项比较。
6. 完成《第48讲_练习题与答案.pdf》，先独立作答再看答案。

运行环境：
- Python 3.x
- NumPy
- PyTorch
- Matplotlib

运行命令：
python lesson48_cnn_basics_experiment.py

本讲最低验收：
- 能解释 (N,C,H,W)。
- 能手算一个 3x3 Kernel 对一个 Patch 的输出。
- 能计算 Conv2d 参数量与输出 Shape。
- 能解释参数共享、多 Channel Feature Map、Max Pooling。
- 能区分 translation equivariance 与 invariance。
- 能跟踪 TinyCNN 从 (N,1,8,8) 到 (N,2) 的完整 Shape。

注意：CNN 在本课程中以“理解即可”为主，不要求深入 ResNet、目标检测、分割或视觉训练工程。
