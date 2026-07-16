# Scripts

Exploration and benchmark scripts. Each phase adds scripts that build on the previous.

## Phase 1 — Run the Black Box

| Script | What it does |
|---|---|
| [`run_qwen3.py`](run_qwen3.py) | Generate text with Qwen3-0.6B via HuggingFace |
| [`param_audit.py`](param_audit.py) | Count parameters, print every tensor shape |
| [`prefill_benchmark.py`](prefill_benchmark.py) | Benchmark prefill at increasing prompt lengths |
| [`decode_benchmark.py`](decode_benchmark.py) | Benchmark decode with and without KV cache |

## Phase 2 — Understand Every Component

| Script | What it does |
|---|---|
| [`inspect_template.py`](inspect_template.py) | Inspect ChatML template, special tokens, token IDs |
| [`inspect_kv_cache.py`](inspect_kv_cache.py) | Inspect HF's `past_key_values` structure and growth |
| [`kv_cache_budget.py`](kv_cache_budget.py) | KV cache memory formula across model sizes |
| [`fragmentation_demo.py`](fragmentation_demo.py) | Simulate pre-alloc vs paged allocation waste |
| [`block_table_demo.py`](block_table_demo.py) | Paged attention block table mapping demo |
| [`trace_attention.py`](trace_attention.py) | Step-by-step attention forward tracer (GQA, scores) |
| [`trace_block.py`](trace_block.py) | One complete transformer block end to end |
| [`verify_rope.py`](verify_rope.py) | Verify RoPE relative-position property |
| [`forward_vs_generate.py`](forward_vs_generate.py) | Compare forward() vs generate() vs manual loop |
| [`simple_engine.py`](simple_engine.py) | Simple inference engine with own sampling + stops |
