《面向 LLM / Agent Systems 科研的深度学习零基础课》
第29讲：Autograd —— loss.backward() 到底怎样自动得到梯度

建议学习顺序：
1. 第29讲_讲义_Autograd自动微分.pdf
2. 第29讲_核心概念速查卡.pdf
3. 第29讲_实验设计_Autograd与梯度累加.pdf
4. lesson29_autograd_experiment.py
5. 第29讲_实验实际输出.txt
6. 第29讲_练习题与答案.pdf

本讲最低目标：
- 知道 requires_grad 只是开启追踪，不是梯度已经存在；
- 理解 leaf / non-leaf / grad_fn / .grad；
- 能解释 loss.backward() = 计算图上的自动 Backprop；
- 知道梯度默认累加，训练 Step 之间需要清梯度；
- 知道 backward 只算梯度，optimizer.step 才更新参数；
- 能把 nn.Linear + MSE 的 Autograd 与手工矩阵梯度对上。

实验脚本说明：
- 只依赖 numpy、torch、matplotlib；
- 会自动打印 Python/NumPy/PyTorch/CUDA 环境；
- 当前材料生成环境为 CPU-only；
- 在本地 CUDA 版 PyTorch 环境运行不影响本讲数学结论；
- 图像输出到 figures/ 目录。
