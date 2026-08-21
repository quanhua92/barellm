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

## Current status

BareLLM can load a Qwen3-compatible checkpoint and generate text using:

- Qwen3 embeddings, RMSNorm, RoPE, SwiGLU, GQA, and tied LM head;
- contiguous KV cache as a reference implementation;
- paged KV cache with fixed-size physical blocks;
- uncached full-sequence recomputation as a correctness reference;
- one-token batched decode with unequal request lengths and padding masks;
- MHA, GQA, and MQA cache-equivalence coverage across contiguous and paged
  storage.

The current paged backend gathers pages into dense tensors before PyTorch SDPA. Direct paged attention is a future optimization.

The current SDPA contract and its prefill/decode masking rules are documented
in [`docs/SDPA.md`](docs/SDPA.md).
The cache protocols, storage backends, block ownership, and request lifecycle
are documented in [`docs/CACHE.md`](docs/CACHE.md).
The engine event stream and generation timing metrics are documented in
[`docs/EVENTS.md`](docs/EVENTS.md).

## Run the demos

Load Qwen3 and benchmark a prefill pass:

```bash
uv run python examples/load_qwen3.py --seq-len 128 --runs 3
```

Generate text with the paged KV cache:

```bash
uv run python examples/generate_demo.py
uv run python examples/generate_demo.py "Say hello world"
uv run python examples/generate_demo.py --no-cache "Say hello world"
uv run python examples/batch_demo.py
```

`generate_demo.py` uses the public `barellm.engine.generate()` API. The lower-
level wiring example is available as:

```bash
uv run python examples/engine_demo.py "Say hello world"
uv run python examples/engine_demo.py --no-cache "Say hello world"
```

The shared device configuration selects CUDA, MPS, or CPU automatically.

The default demo uses the paged KV cache. `--no-cache` recomputes the complete
sequence at every decode step and is intended for correctness comparisons.

The batch demo drives the lower-level engine with multiple requests:

```bash
uv run barellm generate \
  --prompt "Explain paged attention." \
  --max-new-tokens 128
```

## Ownership

- **Scheduler** - owns: request queue | give: resources -> return: who runs
- **KVCacheManager** - owns: request cache lifecycle and block tables
  - **BlockPool** - owns: physical block IDs | give: count -> return: blocks
  - **PagedKVCache** - owns: physical K/V tensors and logical-to-physical writes
- **KVCache view** - model-facing interface for one request or a batch
- **Attention** - owns: attention math | give: hidden + cache view -> return: context

## Scheduler

- Manages request lifecycle: waiting -> running -> finished
- Checks physical capacity through `BlockPool.can_allocate()` before admitting new requests
- Frees blocks when requests finish (EOS, max tokens)

## KV Cache

- Fixed-size blocks (e.g., 16 tokens) scattered across GPU memory
- Each request has a block table mapping logical positions -> physical blocks
- No pre-allocation per request - blocks allocated on demand, freed on finish
- **KVCacheManager** - allocates and releases request cache state
  - **BlockPool** - owns all physical block IDs and tracks free capacity
  - **PagedKVCache** - the actual `[layers, physical_blocks, kv_heads, block_size, head_dim]` K/V tensors
  - **BatchKVCache** - routes each batch row to its request cache and pads unequal histories

```
position -> logical block (pos // block_size) -> block_table -> physical block -> K/V
```

## Development

Run the full checks:

```bash
uv run pytest
uv run pyright src tests
```

The implementation roadmap is in [`docs/ROADMAP.md`](docs/ROADMAP.md).
