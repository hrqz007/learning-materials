# 第3讲：Embedding 几何直觉实验
# 只使用 Python 标准库，不需要安装 numpy / transformers。

import math

# 玩具 Embedding Table：键是 Token ID，值是人为构造的三维向量
embedding_table = {
    104: [0.82, 0.76, 0.10],  # 猫 cat
    205: [0.79, 0.73, 0.15],  # 狗 dog
    806: [0.05, 0.12, 0.91],  # 汽车 car
}

id_to_name = {104: "cat", 205: "dog", 806: "car"}

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def norm(a):
    return math.sqrt(sum(x * x for x in a))

def cosine_similarity(a, b):
    denominator = norm(a) * norm(b)
    if denominator == 0:
        return 0.0
    return dot(a, b) / denominator

print("=== 1. Token ID -> Embedding lookup ===")
for token_id, vector in embedding_table.items():
    print(f"{id_to_name[token_id]:>4} | id={token_id} | vector={vector}")

print("\n=== 2. Cosine similarity ===")
pairs = [(104, 205), (104, 806), (205, 806)]
for a_id, b_id in pairs:
    sim = cosine_similarity(embedding_table[a_id], embedding_table[b_id])
    print(f"cos({id_to_name[a_id]}, {id_to_name[b_id]}) = {sim:.4f}")

print("\n=== 3. Try it yourself ===")
print("把 car 的向量改成 [0.80, 0.72, 0.12] 再运行，观察相似度如何变化。")
print("注意：这些向量是教学用手工数据，不是真实 LLM 学到的 Embedding。")
