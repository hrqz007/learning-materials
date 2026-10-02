第32讲：Dataset、DataLoader 与 Mini-Batch

建议学习顺序：
1. 第32讲_讲义_DatasetDataLoader与MiniBatch.pdf
2. 第32讲_核心概念速查卡.pdf
3. 先独立完成 第32讲_练习题与答案.pdf 中的题目，再看答案
4. 阅读 第32讲_实验设计_DatasetDataLoader与MiniBatch.pdf
5. 运行 lesson32_dataset_dataloader_experiment.py
6. 对照 第32讲_实验实际输出.txt 和 figures/ 下的实验图

运行：
python lesson32_dataset_dataloader_experiment.py

本讲最低验收标准：
- 能解释 Sample / Dataset / DataLoader / Batch / Mini-Batch；
- 能计算 batches per epoch 和总 optimizer steps；
- 知道 shuffle / drop_last 分别改变什么；
- 能写 Dataset + DataLoader + Training Loop；
- 看到 LLM 的 (B,T) / (B,T,D) 时知道 Batch 维从哪里来；
- 不把 DataLoader、Model、Loss、Optimizer 的职责混在一起。
