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
