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
tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, dtype=dtype, local_files_only=True
)
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


n_decode = 128

for prefill_len in [32, 64, 128, 256, 512, 1024, 2048, 4096]:
    ids = torch.randint(0, tokenizer.vocab_size, (1, prefill_len), device=device)

    # 1) Prefill once
    with torch.inference_mode():
        out = model(ids, use_cache=True)
    past_kv = out.past_key_values
    next_token = out.logits[:, -1:].argmax(dim=-1)

    # 2 Warm up
    with torch.inference_mode():
        for _ in range(3):
            out = model(
                next_token,
                past_key_values=past_kv,
                use_cache=True,
            )
            past_kv = out.past_key_values
            next_token = out.logits[:, -1:].argmax(dim=-1)
    # Benchmark with KV Cache
    times = []
    with torch.inference_mode():
        for _ in range(n_decode):
            sync()
            start = time.perf_counter()
            out = model(
                next_token,
                past_key_values=past_kv,
                use_cache=True,
            )
            past_kv = out.past_key_values
            next_token = out.logits[:, -1:].argmax(dim=-1)
            sync()
            times.append((time.perf_counter() - start))
    min_ms = min(times) * 1000
    med_ms = statistics.median(times) * 1000
    stdev_ms = statistics.stdev(times) * 1000

    print(
        f"Decode WITH cache - prefill_len={prefill_len:<4} "
        f"median={med_ms:8.2f}ms "
        f"min={min_ms:8.2f}ms "
        f"stdev={stdev_ms:7.2f}ms "
        f"throughput={1000.0 / med_ms:8.1f} tok/s"
    )

# Benchmark without KV Cache
for prefill_len in [32, 64, 128, 256, 512]:
    ids = torch.randint(0, tokenizer.vocab_size, (1, prefill_len), device=device)

    # 1) Prefill once
    with torch.inference_mode():
        out = model(ids, use_cache=False)
    next_token = out.logits[:, -1:].argmax(dim=-1)

    # 2 Warm up
    with torch.inference_mode():
        for _ in range(3):
            out = model(
                ids,
                use_cache=False,
            )
            next_token = out.logits[:, -1:].argmax(dim=-1)

    # Benchmark without KV Cache
    times = []
    generated = ids.clone()
    with torch.inference_mode():
        for _ in range(n_decode):
            generated = torch.cat([generated, next_token], dim=-1)
            sync()
            start = time.perf_counter()
            out = model(
                generated,
                use_cache=False,
            )
            next_token = out.logits[:, -1:].argmax(dim=-1)
            sync()
            times.append((time.perf_counter() - start))
    min_ms = min(times) * 1000
    med_ms = statistics.median(times) * 1000
    stdev_ms = statistics.stdev(times) * 1000

    print(
        f"Decode WITHOUT cache - prefill_len={prefill_len:<4} "
        f"median={med_ms:8.2f}ms "
        f"min={min_ms:8.2f}ms "
        f"stdev={stdev_ms:7.2f}ms "
        f"throughput={1000.0 / med_ms:8.1f} tok/s"
    )
