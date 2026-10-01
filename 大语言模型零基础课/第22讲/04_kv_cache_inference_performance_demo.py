import math
import numpy as np
import matplotlib.pyplot as plt

# Teaching-only configuration: roughly Llama-like dimensions, but this is NOT a claim about any specific model.
LAYERS = 32
Q_HEADS = 32
HEAD_DIM = 128
BYTES_PER_ELEM = 2   # e.g. BF16/FP16 storage
BATCH = 1

# K and V are both cached.
def kv_cache_bytes(seq_len, kv_heads, batch=BATCH):
    return batch * seq_len * LAYERS * 2 * kv_heads * HEAD_DIM * BYTES_PER_ELEM

def gib(x):
    return x / (1024**3)

lengths = [4096, 8192, 16384, 32768]
configs = {
    'MHA (32 KV heads)': 32,
    'GQA (8 KV heads)': 8,
    'MQA (1 KV head)': 1,
}

print('KV CACHE SIZE DEMO')
print(f'layers={LAYERS}, q_heads={Q_HEADS}, head_dim={HEAD_DIM}, bytes/elem={BYTES_PER_ELEM}, batch={BATCH}')
print()
for n in lengths:
    print(f'Context length: {n:,} tokens')
    for name, kvh in configs.items():
        print(f'  {name:<20s}: {gib(kv_cache_bytes(n, kvh)):6.3f} GiB')
    print()

# A toy latency decomposition. Numbers are deliberately illustrative, not hardware benchmarks.
def toy_ttft_ms(prompt_tokens):
    queue_ms = 20.0
    fixed_ms = 25.0
    prefill_throughput_tokens_per_s = 150_000.0
    return queue_ms + fixed_ms + 1000.0 * prompt_tokens / prefill_throughput_tokens_per_s

print('TOY TTFT DECOMPOSITION (illustrative only)')
for n in [1024, 8192, 32768]:
    print(f'  prompt={n:>6,} tokens -> toy TTFT={toy_ttft_ms(n):7.1f} ms')
print()

tpot_ms = 35.0
for out_tokens in [20, 100, 500]:
    generation_ms = max(0, out_tokens-1) * tpot_ms
    print(f'If TPOT={tpot_ms:.0f} ms, output={out_tokens:>3} tokens -> after-first-token decode time ~ {generation_ms/1000:.2f} s')
print()

# A simple static-batch padding illustration.
outputs = [12, 34, 7, 40]
static_token_slots = len(outputs) * max(outputs)
useful = sum(outputs)
print('STATIC BATCH PADDING TOY EXAMPLE')
print('  output lengths:', outputs)
print('  useful token slots:', useful)
print('  padded token slots:', static_token_slots)
print(f'  toy utilization: {useful/static_token_slots:.1%}')
print('  Continuous batching can reduce this kind of idle waste, although real serving has additional overheads.')

# Plot cache growth.
plt.figure(figsize=(8.6,5.0))
for name, kvh in configs.items():
    ys=[gib(kv_cache_bytes(n,kvh)) for n in lengths]
    plt.plot(np.array(lengths)/1024, ys, marker='o', label=name)
plt.xlabel('Context length (K tokens)')
plt.ylabel('KV cache (GiB, batch=1)')
plt.title('KV cache grows linearly with context length in this simplified formula')
plt.xticks(np.array(lengths)/1024)
plt.legend()
plt.tight_layout()
plt.savefig(r"/mnt/data/LLM_Lesson22_Material_Pack/kv_cache_scaling_demo.png", dpi=180)
print('\nSaved plot:', r"/mnt/data/LLM_Lesson22_Material_Pack/kv_cache_scaling_demo.png")
