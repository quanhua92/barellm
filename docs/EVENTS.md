# Engine events and generation metrics

BareLLM has two callback layers:

- `on_token(token_id, sequence_length)` is the output stream. Return `False`
  to abort the request.
- `on_event(event)` is an observation stream for lifecycle, cache, scheduling,
  and timing information. It does not control execution.

There is deliberately no `on_finish` callback. Completion is represented by
exactly one typed `RequestFinished` event, and the public `generate()` result
also contains the request's final `finish_reason` and `stop_reason`.

## Public API

```python
from barellm.engine import TimingConfig, generate


def observe(event) -> None:
    print(event.kind, event.request_id)


result = generate(
    engine,
    prompt_ids,
    max_new_tokens=128,
    on_token=lambda token_id, count: print(token_id),
    on_event=observe,
    timing=TimingConfig(synchronize_device=True),
)

print(result.metrics.time_to_first_token)
print(result.metrics.decode_tokens_per_second)
```

`generate()` always collects metrics internally. Supplying `on_event` only
adds a user observer; it does not replace the metric collector.

## Event lifecycle

For a normal single request, the important order is:

```text
RequestSubmitted
  EngineStepStart
    CacheAllocated       (cached mode, only when new blocks are added)
    RequestAdmitted
    PrefillStart
      ModelForwardStart/End (phase=prefill)
      SamplingStart/End    (phase=prefill)
    PrefillEnd
    TokenGenerated       (first token)
    DecodeBatchStart     (once per decode iteration)
      ModelForwardStart/End (phase=decode)
      SamplingStart/End    (phase=decode)
    DecodeBatchEnd
    TokenGenerated
    ...
    RequestFinished       (exactly once)
    CacheReleased        (cached mode)
  EngineStepEnd
```

The engine may emit `AdmissionBlocked` when a request cannot currently obtain
enough cache blocks. If no request can make progress, it emits
`EngineStalled` immediately before raising the existing no-progress error.

In batched decode, one `DecodeBatchStart`/`DecodeBatchEnd` pair contains a
tuple of request IDs. Each request still receives its own `TokenGenerated`
and `RequestFinished` events.

`CacheAllocated` is emitted only when the manager grows a request's physical
block table. A successful cache-capacity check that adds zero blocks emits no
event; `block_ids` contains only the newly allocated physical block IDs.

`ModelForwardStart`/`ModelForwardEnd` isolate the model call from the larger
prefill or decode phase. `SamplingStart`/`SamplingEnd` isolate token selection
from model execution. Their `phase`, `request_ids`, and batch metadata make
the distinction visible in a trace without emitting one event per transformer
layer or token.

The concrete event classes are in `barellm.engine.events`. Every event has:

- `kind`: stable `EventKind` value;
- `timestamp`: `time.perf_counter()` seconds;
- `step`: engine step number;
- `request_id`: one request ID where applicable.

## Metrics

`GenerationResult.metrics` is a `GenerationMetrics` instance:

- `prompt_tokens`: number of input tokens;
- `generated_tokens`: number of output tokens;
- `total_seconds`: submission-to-finish time;
- `prefill_seconds`: prompt forward-pass time;
- `prefill_tokens_per_second`: prompt tokens divided by prefill time;
- `time_to_first_token`: submission-to-first-token time, or `None` when no
  token was generated;
- `decode_seconds`: sum of decode forward-pass times;
- `decode_tokens_per_second`: generated tokens after the first divided by
  decode time;
- `inter_token_latency_seconds`: timestamp differences between generated
  tokens;
- `average_inter_token_latency`: convenience average, or `None` for zero or
  one generated token.

Prefill and decode timings synchronize CUDA and MPS by default so asynchronous
device work is included. CPU timing is unaffected. For experiments where
rough timings are sufficient, pass `TimingConfig(synchronize_device=False)`.

The timing boundaries measure model execution and sampling. User code in
`on_token` and `on_event` is not included in the prefill/decode phase duration.
