from typing import cast

import pytest
import torch

from barellm.benchmark import (
    BenchmarkSample,
    check_matching_tokens,
    summarize,
    summarize_samples,
)
from barellm.engine.events import GenerationMetrics


def make_sample(
    mode: str,
    run_index: int,
    wall_seconds: float,
    *,
    ttft: float | None = 0.2,
) -> BenchmarkSample:
    return BenchmarkSample(
        mode=mode,
        run_index=run_index,
        wall_seconds=wall_seconds,
        metrics=GenerationMetrics(
            prompt_tokens=128,
            generated_tokens=4,
            total_seconds=wall_seconds,
            prefill_seconds=0.4,
            prefill_tokens_per_second=320.0,
            time_to_first_token=ttft,
            decode_seconds=0.6,
            decode_tokens_per_second=5.0,
            inter_token_latency_seconds=(0.2, 0.2, 0.2),
        ),
    )


def test_summarize_returns_basic_statistics() -> None:
    assert summarize([1.0, 2.0, 4.0]) == {
        "mean": pytest.approx(7.0 / 3.0),
        "median": 2.0,
        "min": 1.0,
        "max": 4.0,
    }


def test_summarize_rejects_empty_values() -> None:
    with pytest.raises(ValueError, match="empty"):
        summarize([])


def test_summarize_samples_includes_engine_and_wall_metrics() -> None:
    samples = [
        make_sample("cached", 0, 1.0),
        make_sample("cached", 1, 2.0),
    ]

    summary = summarize_samples(samples)
    wall = cast(dict[str, float], summary["wall_seconds"])
    total = cast(dict[str, float], summary["total_seconds"])
    ttft = cast(dict[str, float], summary["time_to_first_token"])
    latency = cast(dict[str, float], summary["inter_token_latency_seconds"])

    assert summary["runs"] == 2
    assert wall["median"] == 1.5
    assert total["max"] == 2.0
    assert ttft["mean"] == 0.2
    assert latency["median"] == 0.2


def test_summarize_samples_omits_missing_optional_metrics() -> None:
    summary = summarize_samples([make_sample("uncached", 0, 1.0, ttft=None)])

    assert "time_to_first_token" not in summary
    assert "inter_token_latency_seconds" in summary


def test_check_matching_tokens_rejects_cached_output_mismatch() -> None:
    reference = torch.tensor([[1, 2]])

    with pytest.raises(RuntimeError, match="different token IDs"):
        check_matching_tokens(reference, torch.tensor([[1, 3]]))
