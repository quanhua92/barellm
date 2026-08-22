# BareLLM server

`barellm serve` starts the small HTTP surface used for health checks and local
profile inspection. It reads configuration from the process environment and an
optional `.env` file in the working directory. A real environment variable
takes precedence over `.env`.

This is a diagnostics surface, not yet the model-serving/OpenAI-compatible API;
future inference routes can live under `/v1` without changing the profile API.

```bash
cp .env.example .env
uv run barellm serve
```

The default server address is `http://0.0.0.0:8000`. The profile API is enabled
by default for local development. Disable it in production when it is not
needed:

```dotenv
BARELLM_ENABLE_PROFILE_API=false
```

## Settings

| Variable | Default | Purpose |
| --- | --- | --- |
| `BARELLM_MODEL_ID` | `Qwen/Qwen3-0.6B` | Default model for examples and CLI generation |
| `BARELLM_DEVICE` | `auto` | `cuda`, `mps`, `cpu`, or automatic detection |
| `BARELLM_DTYPE` | `auto` | `float32`, `float16`, `bfloat16`, or automatic selection |
| `BARELLM_HOST` | `0.0.0.0` | Uvicorn bind address |
| `BARELLM_PORT` | `8000` | Uvicorn listen port |
| `BARELLM_PROFILE_ROOT` | `profiles` | Root directory containing profile runs |
| `BARELLM_ENABLE_PROFILE_API` | `true` | Register profile endpoints and dashboard |

There are intentionally no profile-related server flags. This keeps deployment
configuration in one place and makes disabling the potentially sensitive trace
API explicit through environment configuration.

## Endpoints

- `GET /health` returns `{"status":"ok"}`.
- `GET /api/profiles` discovers timestamped profile runs and older flat runs.
- `GET /api/profiles/{run_id}/metrics` returns `metrics.json`.
- `GET /api/profiles/{run_id}/trace/engine` streams the lightweight engine trace.
- `GET /api/profiles/{run_id}/trace/torch` streams the optional PyTorch trace.
- `GET /profiles` serves a small dashboard.

The dashboard loads the trace into the hosted [Perfetto UI](https://ui.perfetto.dev/)
through its embedded iframe messaging API. The large `torch.trace.json` file is
only fetched when its button is selected; it is not loaded during discovery.

The API exposes files under `BARELLM_PROFILE_ROOT` only, validates discovered
run IDs, and restricts trace names to `engine` and `torch`.
