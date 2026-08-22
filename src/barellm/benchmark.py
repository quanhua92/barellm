"""Small helpers for collecting comparable generation benchmark results."""

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from statistics import mean, median

import torch

from barellm.engine.events import GenerationMetrics


@dataclass(frozen=True)
class BenchmarkSample:
    """One measured generation run."""

    mode: str
    run_index: int
    wall_seconds: float
    metrics: GenerationMetrics

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "run_index": self.run_index,
            "wall_seconds": self.wall_seconds,
            "metrics": asdict(self.metrics),
        }


def summarize(values: Sequence[float]) -> dict[str, float]:
    """Return stable summary statistics for a non-empty sample collection."""
    if not values:
        raise ValueError("cannot summarize an empty collection")
    return {
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def speedup(baseline_seconds: float, candidate_seconds: float) -> float:
    """Return how many times faster a candidate is than a baseline."""
    if baseline_seconds <= 0.0 or candidate_seconds <= 0.0:
        raise ValueError("benchmark times must be positive")
    return baseline_seconds / candidate_seconds


def check_matching_tokens(
    reference: torch.Tensor | None,
    current: torch.Tensor,
) -> torch.Tensor:
    """Validate that a benchmark run matches the first run's token IDs."""
    if reference is not None and not torch.equal(reference, current):
        raise RuntimeError("benchmark runs produced different token IDs")
    return current if reference is None else reference


def compare_token_ids(
    expected: torch.Tensor,
    actual: torch.Tensor,
) -> dict[str, object]:
    """Describe whether two generated token sequences are identical."""
    if expected.shape != actual.shape:
        return {
            "matched": False,
            "reason": "token ID shapes differ",
            "expected_shape": list(expected.shape),
            "actual_shape": list(actual.shape),
        }

    mismatches = torch.nonzero(expected != actual, as_tuple=False)
    if mismatches.numel() == 0:
        return {"matched": True}

    first = mismatches[0]
    position = int(first[-1].item())
    return {
        "matched": False,
        "reason": "token IDs differ",
        "first_mismatch": position,
        "expected_token": int(expected.flatten()[position].item()),
        "actual_token": int(actual.flatten()[position].item()),
    }


def summarize_samples(samples: Sequence[BenchmarkSample]) -> dict[str, object]:
    """Summarize wall-clock and engine metrics from benchmark samples."""
    if not samples:
        raise ValueError("cannot summarize an empty sample collection")

    summary: dict[str, object] = {
        "runs": len(samples),
        "wall_seconds": summarize([sample.wall_seconds for sample in samples]),
    }
    metric_fields = (
        "total_seconds",
        "prefill_seconds",
        "decode_seconds",
        "prefill_tokens_per_second",
        "decode_tokens_per_second",
    )
    for field_name in metric_fields:
        summary[field_name] = summarize(
            [float(getattr(sample.metrics, field_name)) for sample in samples]
        )

    ttft_values = [
        metrics.time_to_first_token
        for metrics in (sample.metrics for sample in samples)
        if metrics.time_to_first_token is not None
    ]
    if ttft_values:
        summary["time_to_first_token"] = summarize(ttft_values)

    latency_values = [
        latency
        for sample in samples
        for latency in sample.metrics.inter_token_latency_seconds
    ]
    if latency_values:
        summary["inter_token_latency_seconds"] = summarize(latency_values)

    return summary
