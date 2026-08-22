import json
from datetime import UTC, datetime

import pytest
import torch

from barellm.engine.events import (
    DecodeBatchEnd,
    DecodeBatchStart,
    EngineStepEnd,
    EngineStepStart,
    GenerationMetrics,
    PrefillEnd,
    PrefillStart,
    RequestFinished,
    RequestSubmitted,
    TokenGenerated,
)
from barellm.engine.profiling import (
    TorchProfiler,
    TraceRecorder,
    profile_run_dir,
)


def make_events() -> list:
    return [
        RequestSubmitted(
            timestamp=10.0,
            step=0,
            request_id="request-1",
            prompt_tokens=3,
        ),
        EngineStepStart(timestamp=10.1, step=1),
        PrefillStart(
            timestamp=10.2,
            step=1,
            request_id="request-1",
            prompt_tokens=3,
            use_cache=True,
        ),
        PrefillEnd(
            timestamp=10.3,
            step=1,
            request_id="request-1",
            prompt_tokens=3,
            use_cache=True,
            duration_seconds=0.1,
        ),
        TokenGenerated(
            timestamp=10.31,
            step=1,
            request_id="request-1",
            token_id=7,
            generated_count=1,
            sequence_length=4,
            is_first_token=True,
        ),
        DecodeBatchStart(
            timestamp=10.4,
            step=2,
            request_ids=("request-1",),
            use_cache=True,
        ),
        DecodeBatchEnd(
            timestamp=10.5,
            step=2,
            request_ids=("request-1",),
            use_cache=True,
            duration_seconds=0.1,
        ),
        RequestFinished(
            timestamp=10.51,
            step=2,
            request_id="request-1",
            finish_reason="length",
            stop_reason=None,
            generated_count=1,
            prompt_tokens=3,
            sequence_length=4,
        ),
        EngineStepEnd(timestamp=10.6, step=2, progressed=True),
    ]


def test_trace_recorder_exports_chrome_trace(tmp_path) -> None:
    recorder = TraceRecorder()
    for event in make_events():
        recorder(event)

    path = recorder.export_chrome_trace(
        tmp_path / "engine.trace.json",
        metadata={"model": "tiny"},
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert path.exists()
    assert payload["traceEvents"][0]["ph"] == "M"
    names = {event["name"] for event in payload["traceEvents"]}
    assert {"engine_step", "prefill", "decode_batch"} <= names
    assert "token_generated" in names
    duration_events = [event for event in payload["traceEvents"] if event["ph"] == "X"]
    assert all(event["dur"] >= 0 for event in duration_events)


def test_trace_slices_use_timestamps_and_do_not_partially_overlap(tmp_path) -> None:
    events = make_events()
    prefill_end = next(event for event in events if isinstance(event, PrefillEnd))
    prefill_end = PrefillEnd(
        timestamp=prefill_end.timestamp,
        step=prefill_end.step,
        request_id=prefill_end.request_id,
        prompt_tokens=prefill_end.prompt_tokens,
        use_cache=prefill_end.use_cache,
        duration_seconds=0.2,
    )
    events[3] = prefill_end

    recorder = TraceRecorder()
    for event in events:
        recorder(event)
    payload = json.loads(
        recorder.export_chrome_trace(tmp_path / "nested.trace.json").read_text(
            encoding="utf-8"
        )
    )
    slices = sorted(
        (event for event in payload["traceEvents"] if event["ph"] == "X"),
        key=lambda event: event["ts"],
    )

    for index, left in enumerate(slices):
        left_end = left["ts"] + left["dur"]
        for right in slices[index + 1 :]:
            right_end = right["ts"] + right["dur"]
            partial_overlap = (
                left["ts"] < right["ts"] < left_end < right_end
                or right["ts"] < left["ts"] < right_end < left_end
            )
            assert not partial_overlap

    prefill = next(event for event in slices if event["name"] == "prefill")
    assert prefill["dur"] == pytest.approx(100_000)
    assert prefill["args"]["end"]["measured_duration_seconds"] == 0.2


def test_trace_recorder_exports_metrics_json(tmp_path) -> None:
    metrics = GenerationMetrics(
        prompt_tokens=3,
        generated_tokens=2,
        total_seconds=0.4,
        prefill_seconds=0.1,
        prefill_tokens_per_second=30.0,
        time_to_first_token=0.2,
        decode_seconds=0.3,
        decode_tokens_per_second=3.33,
        inter_token_latency_seconds=(0.3,),
    )
    recorder = TraceRecorder()

    path = recorder.export_metrics(
        tmp_path / "metrics.json",
        metrics,
        metadata={"device": "cpu"},
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["metadata"] == {"device": "cpu"}
    assert payload["metrics"]["prompt_tokens"] == 3
    assert payload["metrics"]["average_inter_token_latency"] == 0.3


def test_torch_profiler_exports_chrome_trace(tmp_path) -> None:
    path = tmp_path / "torch.trace.json"
    with TorchProfiler(path):
        values = torch.ones(4, 4)
        _ = values + 1

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert path.exists()
    assert payload["traceEvents"]


def test_profile_run_dir_is_timestamped_and_unique(tmp_path) -> None:
    timestamp = datetime(2026, 8, 21, 23, 7, 47, tzinfo=UTC)
    first = profile_run_dir(
        tmp_path,
        model_name="Qwen/Qwen3-0.6B",
        device="mps",
        now=timestamp,
    )
    second = profile_run_dir(
        tmp_path,
        model_name="Qwen/Qwen3-0.6B",
        device="mps",
        now=timestamp,
    )

    assert first.parent.name == "qwen3-0.6b"
    assert first.name == "2026-08-21T23-07-47-mps"
    assert second.name == "2026-08-21T23-07-47-mps-1"
    assert first.is_dir()
    assert second.is_dir()
