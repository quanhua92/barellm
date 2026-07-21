# Journal

Learning notes, appended chronologically.

---

## 2026-07-15 — Project scaffold

Set up BareLLM: uv project, CLI skeleton, 8-phase roadmap, pre-commit hook.
Build order is inside-out — model first, engine next, server last. No engine code yet.

---

## 2026-07-16 — Phase 1 complete

Phase 1 is done — the black box runs and is measured.

**Benchmarks** (`scripts/`):
- Prefill: measured time-to-first-token
- Decode: measured tokens/sec, compared with vs without KV cache

**CLI** (`src/barellm/`):
- `barellm pull/ls/rm` — download, list, and remove models
- `barellm run` — generate text via HuggingFace

Next up: Phase 2 — crack open the black box and trace every operation.

---

## 2026-07-21 — Phase 2 complete

Cracked open the black box. Traced every component from input tokens to output text.

Implemented our own sampler (greedy, temperature, top-k, top-p combined) and stop conditions (OpenAI-compatible finish_reason). Built a simple inference engine that replaces model.generate() with our own prefill + decode loop.

Biggest surprises: head_dim is decoupled (q_proj outputs 2048, not 1024), RoPE has 64 frequency planes spanning period 6 to 5M positions, and pre-allocating KV cache wastes ~70% of memory.

Next up: Phase 3 — replace HuggingFace components with custom implementations.
