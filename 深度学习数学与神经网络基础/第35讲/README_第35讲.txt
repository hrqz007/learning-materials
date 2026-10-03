《面向 LLM / Agent Systems 科研的深度学习零基础课》
第35讲：Batch、Epoch、Step——训练日志里的三个时间尺度到底怎样换算

建议学习顺序：
1. 第35讲_讲义_BatchEpochStep.pdf
2. lesson35_batch_epoch_step_experiment.py
3. 第35讲_实验实际输出.txt
4. figures/ 下的实验图
5. 第35讲_实验设计_BatchEpochStep与训练预算.pdf
6. 第35讲_练习题与答案.pdf
7. 第35讲_核心概念速查卡.pdf

本讲最低验收标准：
- 能根据 Dataset Size、Batch Size、drop_last 计算 batches/epoch；
- 能区分 micro-batch iteration 与 optimizer step；
- 能解释为什么 gradient accumulation 后二者不再一一对应；
- 能计算 global/effective batch size；
- 能把 LLM 训练预算换算成 tokens per update / total tokens seen；
- 看到“10k steps”时会先确认 step 的定义；
- 知道 scheduler/warmup 应该依附于哪个计数器需要看实现；
- 能判断“相同 Epoch”不一定等于“相同优化预算”。

运行实验：
python lesson35_batch_epoch_step_experiment.py

实验不需要 GPU，也不需要下载外部数据。
