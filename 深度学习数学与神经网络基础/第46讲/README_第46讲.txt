《面向 LLM / Agent Systems 科研的深度学习零基础课》
第46讲：BatchNorm

建议学习顺序：
1. 第46讲_讲义_BatchNorm.pdf
2. 第46讲_核心概念速查卡.pdf
3. lesson46_batchnorm_experiment.py
4. 第46讲_实验实际输出.txt
5. 第46讲_实验设计_BatchNorm统计轴与TrainEval.pdf
6. 第46讲_练习题与答案.pdf

本讲最低目标：
- 能解释 BatchNorm1d 对 (N,C) 输入沿哪个轴统计；
- 能手算 mean / variance / normalize / gamma-beta；
- 能区分 train() 与 eval() 下 BatchNorm 的统计来源；
- 能解释 running_mean / running_var 的用途；
- 能解释同一个样本为什么在训练态会依赖 Batch 同伴；
- 能解释 Small Batch 为什么让 Batch Statistics 更不稳定；
- 能解释 BatchNorm2d 对 (N,C,H,W) 沿 N,H,W 统计；
- 能说明现代 Transformer / LLM 为什么更常见 LayerNorm / RMSNorm。

代码运行：
    python lesson46_batchnorm_experiment.py

依赖：
- Python 3.x
- NumPy
- PyTorch
- Matplotlib

实验会在 figures/ 下生成 PNG，并在终端打印所有关键数值。
代码已固定 NumPy 与 PyTorch 随机种子，便于复现。

注意：
- BatchNorm 的 train/eval 行为不同；验证和测试时不要忘记 model.eval()。
- model.eval() 不等于 torch.no_grad()。
- 本讲实验验证机制，不用于证明 BatchNorm 在所有任务上都优于其他 Normalization。
