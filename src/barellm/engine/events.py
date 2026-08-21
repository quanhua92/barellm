"""Typed lifecycle events and timing results for engine execution."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar


class EventKind(str, Enum):
    REQUEST_SUBMITTED = "request_submitted"
    ENGINE_STEP_START = "engine_step_start"
    ENGINE_STEP_END = "engine_step_end"
    REQUEST_ADMITTED = "request_admitted"
    CACHE_ALLOCATED = "cache_allocated"
    CACHE_RELEASED = "cache_released"
    ADMISSION_BLOCKED = "admission_blocked"
    PREFILL_START = "prefill_start"
    PREFILL_END = "prefill_end"
    DECODE_BATCH_START = "decode_batch_start"
    DECODE_BATCH_END = "decode_batch_end"
    TOKEN_GENERATED = "token_generated"
    REQUEST_FINISHED = "request_finished"
    ENGINE_STALLED = "engine_stalled"


@dataclass(frozen=True, kw_only=True)
class EngineEvent:
    """Base type for events emitted by :class:`barellm.engine.Engine`."""

    timestamp: float
    step: int
    request_id: str | None = None
    kind: ClassVar[EventKind]


@dataclass(frozen=True, kw_only=True)
class RequestSubmitted(EngineEvent):
    kind: ClassVar[EventKind] = EventKind.REQUEST_SUBMITTED
    prompt_tokens: int


@dataclass(frozen=True, kw_only=True)
class EngineStepStart(EngineEvent):
    kind: ClassVar[EventKind] = EventKind.ENGINE_STEP_START


@dataclass(frozen=True, kw_only=True)
class EngineStepEnd(EngineEvent):
    kind: ClassVar[EventKind] = EventKind.ENGINE_STEP_END
    progressed: bool


@dataclass(frozen=True, kw_only=True)
class RequestAdmitted(EngineEvent):
    kind: ClassVar[EventKind] = EventKind.REQUEST_ADMITTED
    prompt_tokens: int
    use_cache: bool


@dataclass(frozen=True, kw_only=True)
class CacheAllocated(EngineEvent):
    kind: ClassVar[EventKind] = EventKind.CACHE_ALLOCATED
    block_ids: tuple[int, ...]
    sequence_length: int


@dataclass(frozen=True, kw_only=True)
class CacheReleased(EngineEvent):
    kind: ClassVar[EventKind] = EventKind.CACHE_RELEASED
    block_ids: tuple[int, ...]


@dataclass(frozen=True, kw_only=True)
class AdmissionBlocked(EngineEvent):
    kind: ClassVar[EventKind] = EventKind.ADMISSION_BLOCKED
    sequence_length: int
    reason: str


@dataclass(frozen=True, kw_only=True)
class PrefillStart(EngineEvent):
    kind: ClassVar[EventKind] = EventKind.PREFILL_START
    prompt_tokens: int
    use_cache: bool


@dataclass(frozen=True, kw_only=True)
class PrefillEnd(EngineEvent):
    kind: ClassVar[EventKind] = EventKind.PREFILL_END
    prompt_tokens: int
    use_cache: bool
    duration_seconds: float


@dataclass(frozen=True, kw_only=True)
class DecodeBatchStart(EngineEvent):
    kind: ClassVar[EventKind] = EventKind.DECODE_BATCH_START
    request_ids: tuple[str, ...]
    use_cache: bool


@dataclass(frozen=True, kw_only=True)
class DecodeBatchEnd(EngineEvent):
    kind: ClassVar[EventKind] = EventKind.DECODE_BATCH_END
    request_ids: tuple[str, ...]
    use_cache: bool
    duration_seconds: float


@dataclass(frozen=True, kw_only=True)
class TokenGenerated(EngineEvent):
    kind: ClassVar[EventKind] = EventKind.TOKEN_GENERATED
    token_id: int
    generated_count: int
    sequence_length: int
    is_first_token: bool


@dataclass(frozen=True, kw_only=True)
class RequestFinished(EngineEvent):
    kind: ClassVar[EventKind] = EventKind.REQUEST_FINISHED
    finish_reason: str
    stop_reason: int | str | None
    generated_count: int
    prompt_tokens: int
    sequence_length: int


@dataclass(frozen=True, kw_only=True)
class EngineStalled(EngineEvent):
    kind: ClassVar[EventKind] = EventKind.ENGINE_STALLED
    waiting_ids: tuple[str, ...]
    running_ids: tuple[str, ...]
    reason: str


EventCallback = Callable[[EngineEvent], None]


@dataclass(frozen=True)
class TimingConfig:
    """Controls timing accuracy for asynchronous accelerator devices."""

    synchronize_device: bool = True


@dataclass(frozen=True)
class GenerationMetrics:
    """Timing and throughput measurements for one generated request."""

    prompt_tokens: int
    generated_tokens: int
    total_seconds: float
    prefill_seconds: float
    prefill_tokens_per_second: float
    time_to_first_token: float | None
    decode_seconds: float
    decode_tokens_per_second: float
    inter_token_latency_seconds: tuple[float, ...]

    @property
    def average_inter_token_latency(self) -> float | None:
        if not self.inter_token_latency_seconds:
            return None
        return sum(self.inter_token_latency_seconds) / len(
            self.inter_token_latency_seconds
        )


class MetricsCollector:
    """Collects metrics for one request from the engine event stream."""

    def __init__(self, request_id: str):
        self.request_id = request_id
        self.submitted_at: float | None = None
        self.finished_at: float | None = None
        self.prompt_tokens = 0
        self.prefill_seconds = 0.0
        self.decode_seconds = 0.0
        self.token_timestamps: list[float] = []
        self.generated_tokens = 0

    def on_event(self, event: EngineEvent) -> None:
        if isinstance(event, RequestSubmitted):
            if event.request_id == self.request_id and self.submitted_at is None:
                self.submitted_at = event.timestamp
                self.prompt_tokens = event.prompt_tokens
        elif isinstance(event, PrefillEnd):
            if event.request_id == self.request_id:
                self.prompt_tokens = event.prompt_tokens
                self.prefill_seconds += event.duration_seconds
        elif isinstance(event, DecodeBatchEnd):
            if self.request_id in event.request_ids:
                self.decode_seconds += event.duration_seconds
        elif isinstance(event, TokenGenerated):
            if event.request_id == self.request_id:
                self.token_timestamps.append(event.timestamp)
                self.generated_tokens = event.generated_count
        elif isinstance(event, RequestFinished) and event.request_id == self.request_id:
            self.finished_at = event.timestamp
            self.generated_tokens = event.generated_count

    def build(self, generated_count: int) -> GenerationMetrics:
        prompt_seconds = self.prefill_seconds
        finished_at = self.finished_at
        total_seconds = (
            finished_at - self.submitted_at
            if finished_at is not None and self.submitted_at is not None
            else prompt_seconds + self.decode_seconds
        )
        first_token = self.token_timestamps[0] if self.token_timestamps else None
        ttft = (
            first_token - self.submitted_at
            if (first_token is not None and self.submitted_at is not None)
            else None
        )
        intervals = tuple(
            right - left
            for left, right in zip(
                self.token_timestamps,
                self.token_timestamps[1:],
            )
        )
        decode_token_count = max(generated_count - 1, 0)
        return GenerationMetrics(
            prompt_tokens=self.prompt_tokens,
            generated_tokens=generated_count,
            total_seconds=max(total_seconds, 0.0),
            prefill_seconds=prompt_seconds,
            prefill_tokens_per_second=(
                self.prompt_tokens / prompt_seconds if prompt_seconds > 0 else 0.0
            ),
            time_to_first_token=ttft,
            decode_seconds=self.decode_seconds,
            decode_tokens_per_second=(
                decode_token_count / self.decode_seconds
                if self.decode_seconds > 0
                else 0.0
            ),
            inter_token_latency_seconds=intervals,
        )
