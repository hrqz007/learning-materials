第25讲：两层 MLP 的 Backpropagation

建议学习顺序：
1. 第25讲_讲义_两层MLP的Backpropagation.pdf
2. lesson25_two_layer_mlp_backprop_experiment.py
3. 第25讲_实验实际输出.txt
4. 第25讲_实验设计_两层MLPBackpropagation.pdf
5. 第25讲_练习题与答案.pdf
6. 第25讲_核心概念速查卡.pdf

运行代码：
python lesson25_two_layer_mlp_backprop_experiment.py

本讲最低验收：
- 能写出 Linear backward 三个矩阵公式：dW=X^T@dY、db=sum(dY)、dX=dY@W^T。
- 能解释 ReLU mask 为什么让 W1 某一整列梯度为0。
- 能根据 Shape 判断转置方向。
- 能解释 loss.backward() 在两层 MLP 中自动完成了哪些步骤。
