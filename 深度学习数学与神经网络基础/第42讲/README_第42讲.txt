《面向 LLM / Agent Systems 科研的深度学习零基础课》
第42讲：Train / Validation / Test 与 Data Leakage

建议学习顺序：
1. 第42讲_讲义_TrainValidationTest与DataLeakage.pdf
2. 第42讲_核心概念速查卡.pdf
3. 第42讲_实验设计_TrainValidationTest与DataLeakage.pdf
4. 运行 lesson42_train_val_test_leakage_experiment.py
5. 对照 第42讲_实验实际输出.txt 与 figures/ 中实验图
6. 完成 第42讲_练习题与答案.pdf

本讲最低闭环能力：
- 能严格区分 Train / Validation / Test 的信息权限；
- 能解释为什么 Test 不能反复参与调参；
- 能识别 target/preprocessing/duplicate/group/time/benchmark/memory leakage；
- 能根据科研问题决定 row / group / time / domain split；
- 做 Agent trajectory 实验时，优先先定义“真正独立的 evaluation unit”。

实验依赖：Python 3、NumPy、Matplotlib。
