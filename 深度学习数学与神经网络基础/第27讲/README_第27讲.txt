《面向 LLM / Agent Systems 科研的深度学习零基础课》
第27讲：PyTorch Tensor——从 NumPy 数组进入自动微分框架

建议学习顺序：
1. 第27讲_讲义_PyTorchTensor.pdf
2. lesson27_pytorch_tensor_experiment.py
3. 第27讲_实验实际输出.txt
4. 第27讲_实验设计_PyTorchTensor.pdf
5. 第27讲_练习题与答案.pdf
6. 第27讲_核心概念速查卡.pdf

运行实验：
python lesson27_pytorch_tensor_experiment.py

依赖：
- Python
- NumPy
- PyTorch
- Matplotlib

本讲最低验收：
- 能解释 Tensor 的 shape / dtype / device / requires_grad；
- 能区分 * 与 @；
- 能解释 from_numpy 共享内存与 torch.tensor 复制的差别；
- 能解释 requires_grad=True 为什么不代表已经有 grad；
- 能判断 (B,T,D)@(D,O) -> (B,T,O)。

说明：
当前材料包的验证环境为 PyTorch 2.10.0+cpu，CUDA 不可用。
代码会根据实际机器自动选择 cuda 或 cpu，因此你在有 CUDA 的本机运行时 device 输出可能不同。
