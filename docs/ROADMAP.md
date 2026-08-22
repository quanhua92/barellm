# BareLLM Roadmap

BareLLM is an understanding-first inference engine. Each milestone should be implemented with explicit tensor-shape tests and verified against a simpler reference path before optimizing it.

## Current foundation

- [x] Request lifecycle and stop conditions
- [x] Token sampling with greedy, top-k, and top-p modes
- [x] Scheduler with waiting, running, and finished requests
- [x] Physical KV-cache block pool
- [x] KV-cache admission and block ownership bookkeeping
- [x] Engine prefill/decode orchestration skeleton
- [x] Replace placeholder engine tests with behavioral assertions

## Level 0 — Transformer primitives

- [x] Add `barellm.models`
- [x] Implement token embeddings and tied LM head
- [x] Implement RMSNorm
- [x] Implement rotary positional embeddings
- [x] Implement SwiGLU MLP
- [x] Add shape, dtype, numerical-stability, and parameter-validation tests

## Level 1 — Attention and decoder model

- [x] Implement causal multi-head attention using PyTorch SDPA
- [x] Implement grouped-query attention
- [x] Support MQA through `num_kv_heads=1`
- [x] Verify MHA/GQA equivalence when query and KV head counts match
- [x] Implement transformer blocks with pre-normalization and residuals
- [x] Implement a Qwen3-compatible decoder-only causal language model
- [x] Add final RMSNorm and LM head
- [x] Verify causal masking and full-sequence logits

## Level 2 — Model configuration and weights

- [x] Add device and dtype configuration
- [x] Add HuggingFace snapshot download and local cache management
- [x] Parse and validate model configuration
- [x] Implement safetensors shard loading
- [x] Implement checkpoint key mapping
- [x] Strictly validate missing and unexpected parameters
- [x] Load a small Qwen3-compatible model
- [x] Add model-loading and finite-output tests

## Level 3 — KV-cache correctness

- [x] Define the cache interface between attention and storage
- [x] Implement contiguous single-request KV caching as a reference
- [x] Implement prompt prefill cache writes
- [x] Implement one-token decode cache updates
- [x] Track per-request logical sequence positions
- [x] Verify cached decode against full recomputation
- [x] Extend cache tests to MHA, GQA, and MQA layouts

## Level 4 — Paged KV cache

- [x] Implement `PagedKVCache` storage:
  `[layers, physical_blocks, kv_heads, block_size, head_dim]`
- [x] Implement physical K/V writes and reads
- [x] Implement logical-position to physical-block translation
- [x] Implement block-table gathering
- [x] Support unequal sequence lengths with padding masks
- [x] Test multi-layer and multi-request cache behavior
- [x] Test block exhaustion, freeing, and reuse
- [x] Integrate paged cache with attention

The current paged implementation gathers K/V into dense tensors before SDPA. Direct paged attention is a later optimization.

## Level 5 — Generation and continuous batching

- [x] Add a single-request `generate` convenience API
- [x] Support cached and no-cache generation paths
- [x] Complete batched decode with per-request positions
- [x] Enforce maximum batch size and KV capacity during admission
- [x] Free blocks when requests finish
- [x] Test mixed prompt lengths and generation limits
- [x] Test callbacks, aborts, EOS, stop strings, deadlines, and zero-token requests
- [x] Add typed lifecycle events and generation timing metrics
- [ ] Add end-to-end prompt-to-token tests

The public `generate()` API currently accepts an already-configured `Engine`
and one prompt tensor. It returns a structured `GenerationResult` and uses the
engine's paged cache. A no-cache reference path and tokenizer-level end-to-end
coverage are still planned.

The lifecycle tests cover representative EOS, length, callback, abort,
stop-string, deadline, and zero-token paths; exhaustive combinations and
resource-stress tests remain future work.

The event stream is documented in [`docs/EVENTS.md`](EVENTS.md). `on_token`
remains the streaming and abort hook; request completion is reported through
the typed `RequestFinished` event and `GenerationResult` fields.

`generate(..., use_cache=False)` is the intentionally slow full-recomputation
reference path. The default remains paged cached generation.

## Level 6 — User-facing tooling

- [x] Add public single-request generation example
- [x] Add cached-generation example
- [x] Add batched-engine example
- [x] Add Qwen3 model-loading example
- [x] Add basic `barellm generate` CLI
- [x] Add exportable engine and PyTorch profiling traces
- [x] Add env-configured `barellm serve` health/profile API
- [x] Add local Perfetto profile dashboard
- [x] Document ownership boundaries and tensor shapes
- [x] Document prefill versus decode
- [x] Document contiguous versus paged caching
- [x] Keep README and roadmap synchronized with the implementation

## Performance

- [x] Export engine event traces and generation metrics
- [x] Benchmark uncached versus cached decode
- [x] Benchmark dense gather from paged cache
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
