《面向 LLM / Agent Systems 科研的深度学习零基础课》
第30讲：nn.Module —— 一个真正的 PyTorch 模型怎样组织 Layer、Parameter 与 Forward

建议学习顺序：
1. 第30讲_讲义_nnModule.pdf
2. 第30讲_核心概念速查卡.pdf
3. lesson30_nn_module_experiment.py
4. 第30讲_实验实际输出.txt
5. 第30讲_实验设计_nnModule.pdf
6. 第30讲_练习题与答案.pdf

本讲最低验收：
- 能自己写一个最小 nn.Module；
- 能解释 __init__ 与 forward 的分工；
- 会用 named_parameters() 查看 Parameter 名称、Shape 与参数量；
- 能区分 Parameter / Buffer / 普通 Tensor；
- 能说明 state_dict 保存什么、不保存什么；
- 知道 model(x) 与 forward 的关系；
- 能检查 backward 后 Parameter 是否真正获得 grad。

运行：
python lesson30_nn_module_experiment.py

注意：当前材料生成环境为 CPU-only；代码会自动检测 CUDA。
