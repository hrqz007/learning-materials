第54讲《Scaled Dot-Product Attention》学习顺序

1. 先读完整讲义，务必把每一步 Shape 写在纸上。
2. 独立完成练习题前两页，再看答案。
3. 阅读实验设计，运行 Python 代码。
4. 对照“实验实际输出 TXT”。
5. 最后用速查卡复习。

最低验收标准：
- 能解释 QK^T 每个元素和 (T,T) Shape。
- 能解释为什么除以 sqrt(d_k)。
- 能解释 Causal Mask 为什么在 Softmax 前生效。
- 能解释 Softmax 为什么沿 Key 维，且每行和为 1。
- 能手算 weights @ V。
- 知道 QKV Projection 与真正 Token mixing 的区别。
- 能写出 Batch/Head 版本的 Shape：
  Q,K,V=(B,H,T,d)
  Scores=(B,H,T,T)
  Output=(B,H,T,d_v)

下一讲：
第55讲《Mask 与 Multi-Head Attention》
重点进入 reshape、transpose、多个 Head、concat 和 output projection。
