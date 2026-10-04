《面向 LLM / Agent Systems 科研的深度学习零基础课》
第44讲：Dropout

建议学习顺序：
1. 第44讲_讲义_Dropout.pdf
2. 第44讲_核心概念速查卡.pdf
3. 第44讲_练习题与答案.pdf（先独立做题，再看答案）
4. 第44讲_实验设计_Dropout.pdf
5. lesson44_dropout_experiment.py
6. 第44讲_实验实际输出.txt
7. figures/ 中的实验结果图

本讲最低验收标准：
- 能解释 p 是丢弃概率，q=1-p 是保留概率；
- 能解释 inverted dropout 为什么训练时要除以 q；
- 知道 Dropout 通常保持 Shape、不含可训练 Parameter；
- 能严格区分 model.train()/model.eval() 与 torch.no_grad()；
- 能解释 Dropout Forward 和 Backward 中随机 Mask 的作用；
- 知道 Dropout 太强会导致欠拟合，不能机械认为越大越好；
- 能看懂 Transformer/LLM 中 hidden dropout、attention dropout、LoRA dropout 等配置的基本含义。

运行实验：
    python lesson44_dropout_experiment.py

实验依赖：Python 3、NumPy、PyTorch、Matplotlib。
