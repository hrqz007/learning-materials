第10讲：MLP / Feed-Forward Neural Network

建议学习顺序：
1. 第10讲_讲义_MLP与FeedForwardNetwork.pdf
2. 在纸上手算讲义中的 3 -> 4 -> 2 MLP
3. 第10讲_实验设计_MLP完整Forward与PyTorch对照.pdf
4. 运行 lesson10_mlp_experiment.py
5. 对照 第10讲_实验实际输出.txt
6. 完成 第10讲_练习题与答案.pdf（先做题，后看答案）
7. 最后使用 第10讲_核心概念速查卡.pdf 复习

运行方式：
    python lesson10_mlp_experiment.py

依赖：numpy、matplotlib；PyTorch 对照为可选项，未安装时脚本会自动跳过。

本讲最低验收：
- 能手算 Input -> Linear -> ReLU -> Linear -> Output；
- 能写出每一步 Shape；
- 能计算两层 MLP 参数量；
- 能解释普通 MLP 为什么不负责 Token 间通信；
- 能把 Transformer FFN 还原成 Linear + Activation + Linear 这一结构族。
