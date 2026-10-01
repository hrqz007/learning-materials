《面向 LLM / Agent Systems 科研的深度学习零基础课》
第26讲：NumPy 手写完整 Backpropagation + Training Loop

推荐学习顺序：
1. 第26讲_讲义_NumPy完整Backpropagation与TrainingLoop.pdf
2. lesson26_numpy_training_loop_experiment.py
3. 第26讲_实验实际输出.txt
4. figures/ 下四张实验图
5. 第26讲_实验设计_NumPy完整TrainingLoop.pdf
6. 第26讲_练习题与答案.pdf
7. 第26讲_核心概念速查卡.pdf（最后复习）

运行环境：
- Python 3.x
- numpy
- matplotlib

运行命令：
python lesson26_numpy_training_loop_experiment.py

本讲最低验收：
- 能解释 Forward -> Loss -> Backward -> Update 为什么构成训练闭环；
- 能读懂 dW=X^T@dY、db=sum(dY)、dX=dY@W^T；
- 能说明 Gradient Check 在验证什么；
- 能解释 Learning Rate 太小/适中/太大的差别；
- 明确 Training Loss 下降不等于 Generalization 好。

下一讲：第27讲 PyTorch Tensor——从 NumPy 数组进入自动微分框架。
