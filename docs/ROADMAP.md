# BareLLM Roadmap

BareLLM is an understanding-first inference engine. Each milestone should be implemented with explicit tensor-shape tests and verified against a simpler reference path before optimizing it.

## Current foundation

- [x] Request lifecycle and stop conditions
- [x] Token sampling with greedy, top-k, and top-p modes
- [x] Scheduler with waiting, running, and finished requests
- [x] Physical KV-cache block pool
- [x] KV-cache admission and block ownership bookkeeping
- [x] Engine prefill/decode orchestration skeleton
- [ ] Replace placeholder engine tests with behavioral assertions

## Level 0 — Transformer primitives

- [ ] Add `barellm.models`
- [ ] Implement token embeddings and tied LM head
- [ ] Implement RMSNorm
- [ ] Implement rotary positional embeddings
- [ ] Implement SwiGLU MLP
- [ ] Add shape, dtype, numerical-stability, and parameter-validation tests

## Level 1 — Attention and decoder model

- [ ] Implement causal multi-head attention using PyTorch SDPA
- [ ] Implement grouped-query attention
- [ ] Support MQA through `num_kv_heads=1`
- [ ] Verify MHA/GQA equivalence when query and KV head counts match
- [ ] Implement transformer blocks with pre-normalization and residuals
- [ ] Implement a generic decoder-only causal language model
- [ ] Add final RMSNorm and LM head
- [ ] Verify causal masking and full-sequence logits

## Level 2 — Model configuration and weights

- [ ] Add device and dtype configuration
- [ ] Add HuggingFace snapshot download and local cache management
- [ ] Parse and validate model configuration
- [ ] Implement safetensors shard loading
- [ ] Implement checkpoint key mapping
- [ ] Strictly validate missing and unexpected parameters
- [ ] Load a small Qwen3-compatible model
- [ ] Add model-loading and finite-output tests

## Level 3 — KV-cache correctness

- [ ] Define the cache interface between attention and storage
- [ ] Implement contiguous single-request KV caching as a reference
- [ ] Implement prompt prefill cache writes
- [ ] Implement one-token decode cache updates
- [ ] Track per-request logical sequence positions
- [ ] Verify cached decode against full recomputation
- [ ] Extend cache tests to MHA, GQA, and MQA layouts

## Level 4 — Paged KV cache

- [ ] Implement `PagedKVCache` storage:
  `[layers, physical_blocks, block_size, kv_heads, head_dim]`
- [ ] Implement physical K/V writes and reads
- [ ] Implement logical-position to physical-block translation
- [ ] Implement block-table gathering
- [ ] Complete `KVCacheManager.update_batch`
- [ ] Support unequal sequence lengths with padding masks
- [ ] Test multi-layer and multi-request cache behavior
- [ ] Test block exhaustion, freeing, and reuse
- [ ] Integrate paged cache with attention

The first paged implementation may gather K/V into dense tensors before SDPA. Direct paged attention is a later optimization.

## Level 5 — Generation and continuous batching

- [ ] Add a single-request `generate` convenience API
- [ ] Support cached and no-cache generation paths
- [ ] Complete batched decode with per-request positions
- [ ] Enforce maximum batch size and KV capacity during admission
- [ ] Free blocks when requests finish
- [ ] Test mixed prompt lengths and generation limits
- [ ] Test callbacks, aborts, EOS, stop strings, deadlines, and zero-token requests
- [ ] Add end-to-end prompt-to-token tests

## Level 6 — User-facing tooling

- [ ] Add direct-generation example
- [ ] Add cached-generation example
- [ ] Add batched-engine example
- [ ] Add Qwen3 model-loading example
- [ ] Complete the CLI
- [ ] Document ownership boundaries and tensor shapes
- [ ] Document prefill versus decode
- [ ] Document contiguous versus paged caching
- [ ] Keep README and roadmap synchronized with the implementation

## Performance

- [ ] Benchmark uncached versus cached decode
- [ ] Benchmark dense gather from paged cache
- [ ] Implement direct paged attention
- [ ] Integrate FlashAttention where compatible
- [ ] Add sliding-window attention
- [ ] Add CUDA graph support
- [ ] Add quantized weights and KV cache

## Advanced serving

- [ ] Prefix caching
- [ ] Shared blocks with reference counting
- [ ] Continuous block reuse
- [ ] Preemption and cache eviction
- [ ] Chunked prefill
- [ ] HTTP API with streaming responses
- [ ] Multi-GPU execution
- [ ] Hybrid attention/state-space layers
- [ ] Pluggable sparse-attention backends

## Completion criteria

BareLLM reaches the first full-featured milestone when:

- An internal decoder model produces logits.
- A Qwen3-compatible checkpoint can be loaded.
- MHA, GQA, and MQA work correctly.
- Cached and uncached decoding match within dtype tolerance.
- Paged KV storage works across layers, blocks, and requests.
- Multiple requests decode in one batch.
- Finished requests release their cache blocks.
- Sampling and stopping behavior are covered by tests.
- Examples run from a clean installation.
- Every documented component exists and has a focused test.
