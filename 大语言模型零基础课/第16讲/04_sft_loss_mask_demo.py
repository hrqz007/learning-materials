"""
第16讲配套实验：Chat Template + SFT Loss Mask
不依赖第三方库。
目的：观察“整段上下文都可见，但只让 assistant 答案 token 贡献 loss”。
"""
import math

messages = [
    ("system", "你是一名助教"),
    ("user", "什么是梯度"),
    ("assistant", "梯度表示损失对参数变化的敏感方向"),
]

# 教学用 tokenizer：为了可读性，直接按预先切好的“词/token”列表表示。
tokens = [
    "<system>", "你", "是", "一名", "助教", "</system>",
    "<user>", "什么", "是", "梯度", "</user>",
    "<assistant>", "梯度", "表示", "损失", "对", "参数", "变化", "的", "敏感", "方向", "</assistant>"
]
roles = [
    "system","system","system","system","system","system",
    "user","user","user","user","user",
    "control",
    "assistant","assistant","assistant","assistant","assistant","assistant","assistant","assistant","assistant","assistant"
]

# 常见 SFT 思路：system/user/control 不计 loss；assistant token 才是监督目标。
labels = [tok if role == "assistant" else "IGNORE" for tok, role in zip(tokens, roles)]

print("=== Toy Chat Template ===")
for r, content in messages:
    print(f"<{r}> {content} </{r}>")

print("\n=== Token / Role / Label ===")
print(f"{'idx':>3}  {'token':<14} {'role':<10} {'label'}")
print("-" * 52)
for i, (tok, role, lab) in enumerate(zip(tokens, roles, labels)):
    print(f"{i:>3}  {tok:<14} {role:<10} {lab}")

# 用一组教学概率模拟回答区域的 next-token 概率。
# “before” 代表 SFT 前：正确回答 token 的概率较低；
# “after” 代表 SFT 后：这些 token 的概率提高。
answer_tokens = [tok for tok, role in zip(tokens, roles) if role == "assistant"]
probs_before = [0.18, 0.20, 0.12, 0.22, 0.16, 0.13, 0.28, 0.11, 0.14, 0.30]
probs_after  = [0.62, 0.68, 0.55, 0.70, 0.64, 0.58, 0.75, 0.60, 0.66, 0.80]

def mean_nll(probs):
    return sum(-math.log(p) for p in probs) / len(probs)

print("\n=== Assistant-token probabilities ===")
for tok, p0, p1 in zip(answer_tokens, probs_before, probs_after):
    print(f"{tok:<8} before={p0:.2f}  after={p1:.2f}")

print("\nMean SFT loss before:", round(mean_nll(probs_before), 4))
print("Mean SFT loss after :", round(mean_nll(probs_after), 4))
print("\n结论：SFT 优化会推动理想 assistant token 的条件概率上升，从而降低回答区域的 cross-entropy loss。")
