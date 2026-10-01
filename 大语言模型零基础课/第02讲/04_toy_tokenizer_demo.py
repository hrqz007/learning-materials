"""
Toy tokenizer demo for Lesson 2.
This is NOT the tokenizer of GPT/Gemini/Claude/DeepSeek.
It only demonstrates: vocabulary changes -> token boundaries and token IDs change.
"""

VOCAB = {
    "人工智能": 10,
    "人工": 11,
    "智能": 12,
    "很": 13,
    "强": 14,
    "。": 15,
    "人": 16,
    "工": 17,
    "智": 18,
    "能": 19,
}

def tokenize_longest_match(text, vocab):
    tokens = []
    i = 0
    keys = sorted(vocab, key=len, reverse=True)
    while i < len(text):
        matched = None
        for piece in keys:
            if text.startswith(piece, i):
                matched = piece
                break
        if matched is None:
            matched = text[i]  # fallback for teaching only
            if matched not in vocab:
                vocab[matched] = max(vocab.values(), default=0) + 1
                keys = sorted(vocab, key=len, reverse=True)
        tokens.append(matched)
        i += len(matched)
    return tokens, [vocab[t] for t in tokens]

def show(text, vocab):
    tokens, ids = tokenize_longest_match(text, dict(vocab))
    print("text  :", text)
    print("tokens:", tokens)
    print("ids   :", ids)
    print("count :", len(tokens))
    print()

text = "人工智能很强。"
print("=== Vocabulary A: contains '人工智能' ===")
show(text, VOCAB)

vocab_b = dict(VOCAB)
vocab_b.pop("人工智能")
print("=== Vocabulary B: remove '人工智能' ===")
show(text, vocab_b)

vocab_c = dict(vocab_b)
vocab_c.pop("人工")
vocab_c.pop("智能")
print("=== Vocabulary C: only smaller pieces ===")
show(text, vocab_c)
