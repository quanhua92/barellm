# BareLLM — Agent Notes

## Project

LLM inference engine built from scratch for learning. Serves Qwen3-0.6B.

## Commands

- `just run` / `just serve` / `just pull` / `just ls` / `just rm <model>` — CLI shortcuts
- `just test` — pytest
- `just lint` — ruff check + format
- `uv sync` — install deps

## Structure

Directories earn their existence — no empty packages. See `docs/ROADMAP.md` for the build plan and which package appears in which phase.

## Conventions

- Conventional commits
- Pre-commit hook: ruff check + format on staged .py files
- `src/` layout, pytest pythonpath = `["src"]`
