# BareLLM: A Minimal AI Inference Engine

## Installation
```bash
# Clone the repository
git clone https://github.com/quanhua92/barellm.git
cd barellm

# Install dependencies and set up virtual environment
uv sync

# Configure git hooks (one-time setup for code quality check on commit)
git config core.hooksPath githooks
```

## Structure

- **models/** - "give tokens, return logits" (attention, RoPE, MLP, RMSNorm, transformer)
- **engine/** - "give a prompt, run inference" (generation loop, KV cache, scheduling, batching)
- **sampling/** - "give logits, pick a token" (sampler, stop conditions)
- **config.py** - device/dtype auto-detection
- **hub.py** - HuggingFace model download/cache
- **examples/** - runnable demos (load model, generate text)
- **tests/** - pytest suite

## Ownership

- **Scheduler** - owns: request queue | give: resources -> return: who runs
- **KVCacheManager** - owns: block tables + KV storage
  - **BlockPool** - owns: block IDs | give: count -> return: blocks
  - **PagedKVCache** - owns: K/V tensors | give: block -> return: K/V
- **Attention** - owns: attention math | give: hidden -> return: context

## Scheduler

- Manages request lifecycle: waiting -> running -> finished
- Checks `KVCacheManager.can_allocate()` before admitting new requests
- Frees blocks when requests finish (EOS, max tokens)

## KV Cache

- Fixed-size blocks (e.g., 16 tokens) scattered across GPU memory
- Each request has a block table mapping logical positions -> physical blocks
- No pre-allocation per request - blocks allocated on demand, freed on finish
- **KVCacheManager** - single interface for both scheduler (allocate/free) and attention (update K/V)
  - **BlockPool** - owns all physical block IDs, tracks free capacity
  - **PagedKVCache** - the actual `[L, P, S, H, D]` K/V tensors on GPU

```
position -> logical block (pos // block_size) -> block_table -> physical block -> K/V
```
