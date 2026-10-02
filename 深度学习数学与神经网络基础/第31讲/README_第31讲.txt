《面向 LLM / Agent Systems 科研的深度学习零基础课》
第31讲：PyTorch Training Loop

建议学习顺序：
1. 第31讲_讲义_PyTorchTrainingLoop.pdf
2. 第31讲_核心概念速查卡.pdf
3. 第31讲_实验设计_PyTorchTrainingLoop.pdf
4. 运行 lesson31_pytorch_training_loop_experiment.py
5. 对照 第31讲_实验实际输出.txt 与 figures/ 下图像
6. 最后完成 第31讲_练习题与答案.pdf

本讲最低验收：
- 能不看资料写出 zero_grad -> forward -> loss -> backward -> step；
- 能解释 backward 与 step 的职责差异；
- 知道 PyTorch Gradient 默认累加；
- 知道 no_grad 与 eval 不是一回事；
- 能从训练日志判断“到底有没有发生参数更新”。

实验环境验证：
- Python 3.13.5
- PyTorch 2.10.0+cpu
- 当前验证容器 CUDA available=False

你自己的 CUDA 环境不影响本讲原理；代码不依赖 GPU。
