# Roadmap

BareLLM is built in phases. Each phase builds on the previous one — you can't optimize what you don't understand, and you can't serve what you haven't built.

Think of it as layers of an onion: start from the outside (running the model), peel inward (understand each component), then build outward again (your own engine, your own server, your own distributed system).

---

## Project Structure

Directories earn their existence — nothing is pre-created empty. Each package appears when the first phase that needs it is reached.

| Phase | Package created | What lives there |
|---|---|---|
| 1 | `scripts/` | Exploration and benchmark scripts |
| 2 | `barellm/sampling/` | Token sampling (greedy, temperature, top-k, top-p) |
| 3 | `barellm/model/` | Transformer built from scratch (norm, RoPE, attention, MLP) |
| 4 | `barellm/engine/` | Inference engine (paged KV cache, scheduling, batching) |
| 4 | `barellm/quant/` | Weight and KV cache quantization |
| 5 | `barellm/serve/` | HTTP API (FastAPI, SSE streaming, OpenAI-compatible) |
| 6 | `barellm/distributed/` | Multi-GPU execution (tensor/pipeline parallelism) |

The build order is inside-out: model internals first, then the engine around them, then the server around that. Every directory that exists tells you how far the build has come.

---

## Spinoff — Refusal Fine-Tuning

Parallel to the core engineering phases, we track a spinoff project to train Qwen3-0.6B to recognize context insufficiency and answer with a clean refusal (`"No"`). 

For the complete preparation, benchmarking, training, and validation plan, see the [FINETUNE.md](file:///Users/quan/workspaces/barellm/docs/FINETUNE.md) guide.

---

## Phase 1 — Run the Black Box

**Goal:** Get a real LLM (Qwen3-0.6B) generating text on your machine. Establish the reference you'll rebuild from scratch.

- [x] Run Qwen3-0.6B with HuggingFace transformers
- [x] Count every parameter — know every tensor shape
- [x] Benchmark the prefill phase (time-to-first-token)
- [x] Benchmark the decode phase (tokens per second)
- [x] Compare decode with and without KV cache
- [x] Wire up `barellm pull`, `barellm ls`, and `barellm rm` CLI commands (model management)
- [x] Wire up `barellm run` CLI command (generate text via HuggingFace)

---

## Phase 2 — Understand Every Component

**Goal:** Crack open the black box. Trace every mathematical operation from input tokens to output text.

- [ ] Implement token sampling (greedy, temperature, top-k, top-p)
- [ ] Understand chat templates and special tokens (ChatML format)
- [ ] Inspect HuggingFace's KV cache structure (`past_key_values`)
- [ ] Derive the KV cache memory formula by hand
- [ ] Simulate the memory fragmentation problem
- [ ] Understand the paged attention block table (logical-to-physical mapping)
- [ ] Trace the full attention computation (GQA, scaled dot-product, causal mask)
- [ ] Trace one complete transformer block (RMSNorm -> Attention -> SwiGLU)
- [ ] Verify RoPE properties (rotary position embedding, relative position)
- [ ] Implement all stop conditions (EOS, max tokens, stop strings, deadline)
- [ ] Compare `model.forward()` vs `model.generate()`
- [ ] Build a simple single-sequence inference engine

---

## Phase 3 — Replace Every Component

**Goal:** Swap each HuggingFace component for your own code. Verify output matches the reference exactly.

- [ ] Write a custom generation loop (prefill + decode with KV cache)
- [ ] Load HuggingFace weights into a custom model
- [ ] Implement RMSNorm from scratch
- [ ] Implement RoPE (rotary position embedding) from scratch
- [ ] Implement grouped-query attention (GQA) from scratch
- [ ] Implement SwiGLU MLP from scratch
- [ ] Assemble a complete transformer from your components
- [ ] Verify custom `forward()` matches HuggingFace output
- [ ] Simulate FlashAttention tiling in PyTorch
- [ ] Add QK-norm (query/key normalization inside attention)

---

## Phase 4 — Optimize for Throughput

**Goal:** Make the engine faster and more memory-efficient. This is where real serving techniques begin.

- [ ] Implement INT8 weight quantization (W8A16)
- [ ] Implement W4A16 group quantization (4-bit weights)
- [ ] Implement dequantize-then-matmul for quantized inference
- [ ] Quantize the KV cache to INT8
- [ ] Analyze padding waste in static batching
- [ ] Implement continuous batching (iteration-level scheduling)
- [ ] Implement chunked prefill (mix prefill + decode in one batch)
- [ ] Build a paged KV cache block manager (allocate, free, refcount)
- [ ] Implement copy-on-write for shared prefix blocks
- [ ] Build a prefix cache using a radix tree
- [ ] Capture and replay CUDA graphs for decode

---

## Phase 5 — Serve Over HTTP

**Goal:** Expose the engine as a real API that clients can call.

- [ ] Build FastAPI endpoints with a request queue
- [ ] Implement SSE (server-sent events) token streaming
- [ ] Handle client disconnect / cancellation
- [ ] Implement OpenAI-compatible API (`/v1/chat/completions`)
- [ ] Add request validation and error handling
- [ ] Add observability (tracing and metrics)
- [ ] Containerize with Docker
- [ ] Wire up `barellm serve` CLI command

---

## Phase 6 — Scale Across GPUs

**Goal:** Run models too large for one GPU. Understand distributed inference.

- [ ] Implement tensor parallelism (column/row sharding + NCCL)
- [ ] Implement pipeline parallelism (micro-batching)
- [ ] Implement distributed data parallel (DDP) gradient sync
- [ ] Implement ZeRO memory partitioning
- [ ] Implement speculative decoding (draft model + verification)
- [ ] Implement mixture of experts (MoE) routing and serving

---

## Phase 7 — Production

**Goal:** Deploy, monitor, and operate the system for real use.

- [ ] Analyze and optimize serving cost (cost per token)
- [ ] Implement autoscaling
- [ ] Build an evaluation pipeline (quality metrics)
- [ ] Deploy to cloud (GPU instance provisioning)

---

## Phase 8 — Capstone

**Goal:** Prove it works. Benchmark, document, and demo.

- [ ] Benchmark BareLLM vs HuggingFace vs vLLM
- [ ] Write the final project README
- [ ] Build an interactive demo (web chat UI)
- [ ] Write a retrospective (what worked, what didn't)
