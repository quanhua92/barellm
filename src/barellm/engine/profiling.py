"""Optional engine trace and PyTorch operator profiling."""

import json
import re
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Self

import torch

from barellm.engine.events import (
    DecodeBatchEnd,
    DecodeBatchStart,
    EngineEvent,
    EngineStepEnd,
    EngineStepStart,
    GenerationMetrics,
    ModelForwardEnd,
    ModelForwardStart,
    PrefillEnd,
    PrefillStart,
    SamplingEnd,
    SamplingStart,
)

_PHASE_PAIRS = (
    (EngineStepStart, EngineStepEnd, "engine_step"),
    (PrefillStart, PrefillEnd, "prefill"),
    (DecodeBatchStart, DecodeBatchEnd, "decode_batch"),
    (ModelForwardStart, ModelForwardEnd, "model_forward"),
    (SamplingStart, SamplingEnd, "sampling"),
)


def profile_run_dir(
    root: str | Path = "profiles",
    *,
    model_name: str = "qwen3",
    device: str | None = None,
    now: datetime | None = None,
) -> Path:
    """Create and return a unique timestamped directory for one profile run."""
    model_slug = (
        re.sub(
            r"[^a-zA-Z0-9._-]+",
            "-",
            model_name.rsplit("/", maxsplit=1)[-1],
        )
        .strip("-")
        .lower()
    )
    timestamp = (now or datetime.now().astimezone()).strftime("%Y-%m-%dT%H-%M-%S")
    device_suffix = f"-{device}" if device else ""
    base_path = Path(root) / model_slug / f"{timestamp}{device_suffix}"
    output_path = base_path
    counter = 1
    while output_path.exists():
        output_path = base_path.with_name(f"{base_path.name}-{counter}")
        counter += 1
    output_path.mkdir(parents=True)
    return output_path


class TraceRecorder:
    """Record engine events and export them as Chrome Trace JSON."""

    def __init__(self) -> None:
        self.events: list[EngineEvent] = []

    def __call__(self, event: EngineEvent) -> None:
        self.events.append(event)

    def on_event(self, event: EngineEvent) -> None:
        self(event)

    def clear(self) -> None:
        self.events.clear()

    def export_chrome_trace(
        self,
        path: str | Path,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> Path:
        """Write a Perfetto/Chrome Trace-compatible engine timeline."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        trace_events = self._build_trace_events(metadata)
        output_path.write_text(
            json.dumps({"traceEvents": trace_events}, indent=2) + "\n",
            encoding="utf-8",
        )
        return output_path

    def export_metrics(
        self,
        path: str | Path,
        metrics: GenerationMetrics,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> Path:
        """Write generation metrics and optional run metadata as JSON."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metrics": metrics.to_dict(),
            "metadata": dict(metadata or {}),
        }
        output_path.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        return output_path

    def _build_trace_events(
        self,
        metadata: Mapping[str, object] | None,
    ) -> list[dict[str, object]]:
        if not self.events:
            trace_events: list[dict[str, object]] = []
            if metadata:
                trace_events.append(self._metadata_event(metadata))
            return trace_events

        base_timestamp = min(event.timestamp for event in self.events)
        trace_events = []
        if metadata:
            trace_events.append(self._metadata_event(metadata))

        open_phases: dict[tuple[str, object], EngineEvent] = {}
        for event in self.events:
            handled_as_phase = False
            for start_type, end_type, name in _PHASE_PAIRS:
                phase_key = (name, self._phase_key(event))
                if isinstance(event, start_type):
                    open_phases[phase_key] = event
                    handled_as_phase = True
                    break
                if isinstance(event, end_type):
                    start_event = open_phases.pop(phase_key, None)
                    if start_event is not None:
                        trace_events.append(
                            self._duration_event(
                                name,
                                start_event,
                                event,
                                base_timestamp,
                            )
                        )
                    handled_as_phase = True
                    break
            if handled_as_phase:
                continue

            trace_events.append(self._instant_event(event, base_timestamp))

        return trace_events

    def _phase_key(self, event: EngineEvent) -> object:
        if isinstance(event, (PrefillStart, PrefillEnd)):
            return event.request_id
        if isinstance(event, (DecodeBatchStart, DecodeBatchEnd)):
            return event.request_ids
        if isinstance(
            event,
            (
                ModelForwardStart,
                ModelForwardEnd,
                SamplingStart,
                SamplingEnd,
            ),
        ):
            return (event.phase, event.request_ids)
        return "engine"

    def _duration_event(
        self,
        name: str,
        start: EngineEvent,
        end: EngineEvent,
        base_timestamp: float,
    ) -> dict[str, object]:
        # Chrome Trace slices must use the event timestamps. A measured
        # duration can include device synchronization that happened between
        # emitting the start and end events, which can otherwise create
        # partially overlapping slices on one Perfetto track.
        duration = max(end.timestamp - start.timestamp, 0.0)
        measured_duration = getattr(end, "duration_seconds", None)
        end_args = self._event_args(end)
        if measured_duration is not None:
            end_args["measured_duration_seconds"] = float(measured_duration)
        return {
            "name": name,
            "cat": "barellm.engine",
            "ph": "X",
            "pid": 1,
            "tid": 0,
            "ts": (start.timestamp - base_timestamp) * 1_000_000,
            "dur": duration * 1_000_000,
            "args": {
                "start": self._event_args(start),
                "end": end_args,
            },
        }

    def _instant_event(
        self,
        event: EngineEvent,
        base_timestamp: float,
    ) -> dict[str, object]:
        return {
            "name": event.kind.value,
            "cat": "barellm.engine",
            "ph": "i",
            "s": "t",
            "pid": 1,
            "tid": 0,
            "ts": (event.timestamp - base_timestamp) * 1_000_000,
            "args": self._event_args(event),
        }

    def _metadata_event(
        self,
        metadata: Mapping[str, object],
    ) -> dict[str, object]:
        return {
            "name": "BareLLM profile",
            "cat": "barellm.metadata",
            "ph": "M",
            "pid": 1,
            "tid": 0,
            "args": dict(metadata),
        }

    def _event_args(self, event: EngineEvent) -> dict[str, object]:
        if not is_dataclass(event):
            return {}
        payload = asdict(event)
        payload.pop("timestamp", None)
        payload.pop("step", None)
        payload.pop("request_id", None)
        payload["request_id"] = event.request_id
        payload["step"] = event.step
        return payload


class TorchProfiler:
    """Export an optional PyTorch CPU/CUDA operator trace.

    CPU activity is always collected. CUDA activity is added when CUDA is
    available. MPS has no profiler activity in the supported PyTorch build,
    so MPS users should combine this with :class:`TraceRecorder`.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        record_shapes: bool = False,
        profile_memory: bool = False,
        with_stack: bool = False,
    ) -> None:
        self.path = Path(path)
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self._profiler: Any = None

    def __enter__(self) -> Self:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        self._profiler = torch.profiler.profile(
            activities=activities,
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack,
        )
        self._profiler.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool | None:
        if self._profiler is None:
            return None
        result = self._profiler.__exit__(exc_type, exc_value, traceback)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._profiler.export_chrome_trace(str(self.path))
        return result
