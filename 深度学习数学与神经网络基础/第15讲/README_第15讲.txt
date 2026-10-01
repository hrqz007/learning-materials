第15讲《MSE：为什么要平方？为什么要平均？》学习顺序

建议顺序：
1. 第15讲_讲义_MSE.pdf
2. 运行 lesson15_mse_experiment.py
3. 查看 第15讲_实验实际输出.txt 和 figures 文件夹
4. 阅读 第15讲_实验设计_MSE与BatchReduction.pdf
5. 完成 第15讲_练习题与答案.pdf（建议先遮住答案）
6. 最后用 第15讲_核心概念速查卡.pdf 复习

运行命令：
python lesson15_mse_experiment.py

依赖：
- Python 3
- NumPy
- Matplotlib
- PyTorch（仅实验8交叉验证；未安装时前7个实验仍可运行）

最低验收：
- 能手算一个小 Batch 的 MSE。
- 能解释为什么平方会放大大误差。
- 能解释 mean 与 sum 的区别。
- 能根据 Tensor Shape 说明 MSE 在哪些元素上求平均。
- 知道 GPT next-token 训练主要不是 MSE，而是后续要学的 Cross Entropy。
