# Profiling BareLLM

BareLLM exposes two complementary profiling layers:

- `TraceRecorder` records engine lifecycle events and exports a dependency-free
  Chrome Trace JSON timeline.
- `TorchProfiler` wraps `torch.profiler` and exports PyTorch operator events.

Both are opt-in. Normal generation does not retain event history or start a
profiler.

## Qwen3 demo

```bash
uv run python examples/profile_demo.py "Explain paged KV caching."
```

The demo creates a timestamped directory such as
`profiles/qwen3-0.6b/2026-08-21T23-07-47-mps/` and writes three files there:

- `engine.trace.json`: BareLLM request, scheduling, prefill, decode, token,
  cache, and finish events;
- `torch.trace.json`: PyTorch operator trace;
- `metrics.json`: `GenerationMetrics` plus run metadata.

The regular generation demos and CLI keep profiling disabled by default. Opt
in with `--profile` for the lightweight engine trace and metrics. Add
`--torch-profile` only when the larger PyTorch operator trace is needed. Use
`--profile-dir` when an exact output directory is preferred:

```bash
uv run python examples/generate_demo.py --profile "Say hello world"
uv run python examples/engine_demo.py --profile "Say hello world"
uv run barellm generate --profile --prompt "Say hello world"
uv run python examples/generate_demo.py --profile --torch-profile "Say hello"
uv run python examples/generate_demo.py --profile-dir profiles/debug "Say hello"
```

Open either trace JSON file with [Perfetto](https://ui.perfetto.dev/) or a
Chrome-compatible trace viewer. The engine trace explains *what the inference
engine was doing*; the PyTorch trace explains *which operators consumed time*.

## Python API

```python
from barellm.engine import TorchProfiler, TraceRecorder, generate

recorder = TraceRecorder()

with TorchProfiler("profiles/run.torch.json"):
    result = generate(
        engine,
        token_ids,
        on_event=recorder,
    )

recorder.export_chrome_trace("profiles/run.engine.json")
recorder.export_metrics("profiles/run.metrics.json", result.metrics)
```

`TraceRecorder` is an `on_event` callback, so it observes the same typed event
stream used for request debugging. Its duration events cover engine steps,
prefill, and decode batches. Instant events mark admission, cache ownership,
generated tokens, request completion, and stalls.

Chrome Trace slice durations use the emitted start/end timestamps so nested
engine, model-forward, and sampling spans remain valid on Perfetto tracks.
Device-synchronized measurements remain available in the slice arguments as
`measured_duration_seconds`.

## Interpreting the files

- Use `metrics.json` for stable aggregate values such as TTFT, prefill
  throughput, decode throughput, and inter-token latency.
- Use `engine.trace.json` to find queue/admission delays, cache growth and
  release, uneven batch behavior, and the exact request lifecycle.
- Use `torch.trace.json` to investigate model operators, tensor shapes, memory,
  and CPU/CUDA kernel activity when the relevant options are enabled.

CUDA activity is included automatically when CUDA is available. The supported
PyTorch build does not expose an MPS profiler activity, so MPS runs still get
the complete BareLLM engine trace and metrics, plus CPU-side PyTorch operator
events.

Profiling adds overhead. Keep `record_shapes`, `profile_memory`, and
`with_stack` disabled unless that extra detail is needed. For accurate engine
timings, keep the default synchronized timing; for lower-overhead experiments,
pass `TimingConfig(synchronize_device=False)` to generation.
