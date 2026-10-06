第55讲《Mask 与 Multi-Head Attention》学习顺序

1. 先读完整讲义，重点画出 Shape 变化。
2. 独立完成练习题。
3. 阅读实验设计，运行 Python 实验。
4. 对照实际输出。
5. 最后用速查卡复习。

最低验收标准：
- 知道 Head 拆的是 feature dimension，不是 Token。
- 能从 (B,T,D) 推到 (B,H,T,d_head)。
- 知道 scaling 用 sqrt(d_head)。
- 能解释 causal mask 与 padding mask 的区别与广播。
- 能追踪 per-head scores / weights / outputs。
- 能把多个 Head concat 回 (B,T,D)。
- 能解释 W_O 的作用。
- 知道 D 固定时增加 Head 数通常不增加标准 QKV/O 参数量。
- 知道 Attention Heatmap 不能单独当因果证明。

下一讲：
第56讲《Transformer 中的 MLP、Residual、LayerNorm / RMSNorm》
把 Multi-Head Attention 重新装回完整 Transformer Block。
