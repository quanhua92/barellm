import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen3-0.6B"

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

if device == "cuda":
    dtype = torch.bfloat16
elif device == "mps":
    dtype = torch.float16
else:
    dtype = torch.float32

print(f"dtype={dtype} device={device}")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
model = model.to(device)
model.eval()
print(f"attn_impl={model.config._attn_implementation}")
if device == "cuda":
    print(
        f"sdpa backends: flash={torch.backends.cuda.flash_sdp_enabled()} "
        f"mem_efficient={torch.backends.cuda.mem_efficient_sdp_enabled()} "
        f"math={torch.backends.cuda.math_sdp_enabled()} "
        f"cudnn={torch.backends.cuda.cudnn_sdp_enabled()}"
    )


def sync():
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


for prompt_len in [32, 64, 128, 256, 512, 1024, 2048, 4096]:
    inputs = torch.randint(0, tokenizer.vocab_size, (1, prompt_len)).to(device)

    # Warm up
    with torch.inference_mode():
        _ = model(inputs, use_cache=True)

    # Benchmark
    times = []
    for _ in range(5):
        sync()

        start = time.perf_counter()
        with torch.inference_mode():
            _ = model(inputs, use_cache=True)

        sync()
        times.append((time.perf_counter() - start))

    min_ms = min(times) * 1000
    med_ms = statistics.median(times) * 1000
    stdev_ms = statistics.stdev(times) * 1000

    print(
        f"Prefill prompt_len={prompt_len:<4} "
        f"median={med_ms:8.2f}ms "
        f"min={min_ms:8.2f}ms "
        f"stdev={stdev_ms:7.2f}ms "
        f"throughput={prompt_len / (med_ms / 1000):8.1f} tok/s"
    )
