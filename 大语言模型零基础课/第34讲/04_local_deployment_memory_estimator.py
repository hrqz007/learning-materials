# 第34讲实验：本地部署纯权重内存估算器
# 只估算模型权重，不代表完整运行显存。
# 真实部署还需要 KV cache、workspace、量化 metadata、runtime buffers 等。

FORMATS = {
    "FP32": 4.0,
    "BF16/FP16": 2.0,
    "INT8": 1.0,
    "ideal INT4": 0.5,
}

MODELS_B = [1, 7, 14, 32]

def weight_gb(params_billion, bytes_per_param):
    return params_billion * 1e9 * bytes_per_param / 1e9

print("Toy local deployment weight-memory estimator")
print("=" * 78)
print("Pure weights only; decimal GB for teaching.")
print()

header = "Model".ljust(8) + "".join(name.rjust(16) for name in FORMATS)
print(header)
print("-" * len(header))

for b in MODELS_B:
    row = f"{b}B".ljust(8)
    for name, byte in FORMATS.items():
        row += f"{weight_gb(b, byte):15.1f}G"
    print(row)

print()
print("Deployment stack checklist")
print("- model architecture/config")
print("- checkpoint/weights")
print("- tokenizer + chat template")
print("- inference runtime")
print("- precision / quantization")
print("- device RAM/VRAM")
print("- context length + KV cache")
print("- batching / concurrency")
print("- application or API layer")

print()
print("Lesson:")
print("- Model size in GB is not the same as runtime memory.")
print("- The same checkpoint can be served by different runtimes.")
print("- Reproducibility requires pinning both model revision and runtime settings.")
