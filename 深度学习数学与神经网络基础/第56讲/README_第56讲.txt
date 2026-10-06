第56讲《Transformer 中的 MLP、Residual、LayerNorm / RMSNorm》学习顺序

1. 先读完整讲义，画出 Pre-Norm Block。
2. 独立完成练习题。
3. 阅读实验设计并运行 Python。
4. 对照实际输出。
5. 最后用速查卡复习。

最低验收标准：
- 能写出 Pre-Norm 两条残差公式。
- 知道 Attention 负责 token mixing，MLP 主要做 token-wise feature transformation。
- 能追踪 D -> D_ff -> D。
- 能解释 Residual identity path 与 dy/dx = I + dF/dx 的直觉。
- 能区分 Pre-Norm 与 Post-Norm。
- 知道 MHA/MLP 在 Add 前都必须回到 (B,T,D)。
- 能粗估 MHA≈4D^2、MLP≈2D*D_ff 的权重参数量。
- 知道 LayerNorm/RMSNorm 与 Dropout 的 train/eval 行为不同。

下一讲：
第57讲《完整 Transformer Block》
从 Token IDs / Embedding 开始，把位置机制、Attention、MLP、Residual、Norm 全部连起来并追踪 Shape。
