import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)
weights = rng.normal(0, 1.4, size=2000)
# Add a few outliers to make the quantization trade-off visible.
weights[:8] = np.array([6.0, -5.5, 4.8, -4.2, 3.9, -3.7, 5.2, -4.9])

def symmetric_quantize(x, bits):
    qmax = 2**(bits-1) - 1
    max_abs = np.max(np.abs(x))
    scale = max_abs / qmax if max_abs > 0 else 1.0
    q = np.clip(np.round(x / scale), -qmax, qmax).astype(np.int32)
    x_hat = q.astype(np.float64) * scale
    return q, x_hat, scale

print('UNIFORM SYMMETRIC QUANTIZATION TOY DEMO')
for bits in [8, 4, 3]:
    q, x_hat, scale = symmetric_quantize(weights, bits)
    mse = np.mean((weights - x_hat)**2)
    mae = np.mean(np.abs(weights - x_hat))
    print(f'{bits}-bit: scale={scale:.5f}, MSE={mse:.6f}, MAE={mae:.6f}, unique_levels_used={len(np.unique(q))}')

print('\nTHEORETICAL WEIGHT STORAGE FOR A 70B-PARAMETER MODEL')
params = 70_000_000_000
for name, bytes_per_param in [('FP32',4), ('FP16/BF16',2), ('INT8',1), ('INT4 ideal',0.5)]:
    gb10 = params * bytes_per_param / 1e9
    gib = params * bytes_per_param / (1024**3)
    print(f'{name:<10s}: {gb10:6.1f} GB decimal  |  {gib:6.1f} GiB binary')
print('Note: real quantized checkpoints/runtime memory include scales, metadata, packing/alignment and other tensors.')

print('\nMOE TOY ACCOUNTING')
experts = 8
expert_params_b = 1.0
shared_params_b = 2.0
top_k = 2
total_b = shared_params_b + experts * expert_params_b
active_b = shared_params_b + top_k * expert_params_b
print(f'experts={experts}, expert_size={expert_params_b}B, shared={shared_params_b}B, top_k={top_k}')
print(f'total parameters (toy)  = {total_b:.1f}B')
print(f'active parameters/token = {active_b:.1f}B (toy approximation)')
print('Important: active parameters are NOT an exact FLOP count; attention, router and communication overhead still matter.')

# Plot original vs reconstructed weights.
_, w8, _ = symmetric_quantize(weights, 8)
_, w4, _ = symmetric_quantize(weights, 4)
idx = np.arange(120)
plt.figure(figsize=(9,5.2))
plt.plot(idx, weights[:120], marker='o', markersize=2.2, linewidth=1.0, label='Original FP-like weights')
plt.plot(idx, w8[:120], linewidth=1.0, label='8-bit reconstruction')
plt.plot(idx, w4[:120], linewidth=1.0, label='4-bit reconstruction')
plt.xlabel('Weight index (first 120)')
plt.ylabel('Value')
plt.title('Lower-bit quantization uses fewer representable levels')
plt.legend()
plt.tight_layout()
plt.savefig(r"/mnt/data/LLM_Lesson23_Material_Pack/quantization_demo.png", dpi=180)
print('\nSaved plot:', r"/mnt/data/LLM_Lesson23_Material_Pack/quantization_demo.png" )
