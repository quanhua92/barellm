"""Compare cached and uncached Qwen3 generation.

Usage:
    uv run python examples/benchmark_generation.py
    uv run python examples/benchmark_generation.py --seq-lens 128,512 --runs 3
    uv run python examples/benchmark_generation.py --output benchmarks/results.json
"""

import argparse
import json
import time
from pathlib import Path
from typing import cast

import torch
from transformers import AutoTokenizer

from barellm.benchmark import (
    BenchmarkSample,
    check_matching_tokens,
    compare_token_ids,
    speedup,
    summarize_samples,
)
from barellm.config import DEVICE, DTYPE, MODEL_ID
from barellm.engine.events import TimingConfig
from barellm.engine.generate import GenerationResult, generate
from barellm.runtime import load_qwen3_engine


def parse_seq_lens(value: str) -> list[int]:
    """Parse a comma-separated list of positive sequence lengths."""
    try:
        sequence_lengths = [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "sequence lengths must be comma-separated integers"
        ) from exc
    if not sequence_lengths or any(length <= 0 for length in sequence_lengths):
        raise argparse.ArgumentTypeError("sequence lengths must be positive")
    return sequence_lengths


def build_prompt(token_ids: torch.Tensor, sequence_length: int) -> torch.Tensor:
    """Repeat a tokenized prompt to create a deterministic target length."""
    repeats = (sequence_length + token_ids.shape[1] - 1) // token_ids.shape[1]
    return token_ids.repeat(1, repeats)[:, :sequence_length].contiguous()


def run_once(
    engine,
    token_ids: torch.Tensor,
    *,
    use_cache: bool,
    max_new_tokens: int,
    mode: str,
    run_index: int,
) -> tuple[GenerationResult, BenchmarkSample]:
    started_at = time.perf_counter()
    result = generate(
        engine,
        token_ids,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        eos_ids=set(),
        use_cache=use_cache,
        request_id=f"benchmark-{mode}-{run_index}",
        timing=TimingConfig(synchronize_device=True),
    )
    wall_seconds = time.perf_counter() - started_at
    return result, BenchmarkSample(
        mode=mode,
        run_index=run_index,
        wall_seconds=wall_seconds,
        metrics=result.metrics,
    )


def benchmark_case(
    engine,
    token_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    runs: int,
    warmup: int,
    sequence_length: int,
) -> dict[str, object]:
    samples: dict[str, list[BenchmarkSample]] = {"cached": [], "uncached": []}
    reference_tokens: dict[str, torch.Tensor | None] = {
        "cached": None,
        "uncached": None,
    }
    repeatability: dict[str, dict[str, object]] = {
        "cached": {"matched": True},
        "uncached": {"matched": True},
    }

    for use_cache, mode in ((True, "cached"), (False, "uncached")):
        for warmup_index in range(warmup):
            result, _ = run_once(
                engine,
                token_ids,
                use_cache=use_cache,
                max_new_tokens=max_new_tokens,
                mode=f"{mode}-warmup",
                run_index=warmup_index,
            )
            if result.generated_count != max_new_tokens:
                raise RuntimeError(
                    f"{mode} warmup generated {result.generated_count} tokens; "
                    f"expected {max_new_tokens}"
                )

        for run_index in range(runs):
            result, sample = run_once(
                engine,
                token_ids,
                use_cache=use_cache,
                max_new_tokens=max_new_tokens,
                mode=mode,
                run_index=run_index,
            )
            if result.generated_count != max_new_tokens:
                raise RuntimeError(
                    f"{mode} generated {result.generated_count} tokens; "
                    f"expected {max_new_tokens}"
                )
            current_tokens = result.token_ids.detach().cpu()
            try:
                reference_tokens[mode] = check_matching_tokens(
                    reference_tokens[mode],
                    current_tokens,
                )
            except RuntimeError as exc:
                repeatability[mode] = {
                    "matched": False,
                    "reason": str(exc),
                }
            samples[mode].append(sample)

    cached_tokens = reference_tokens["cached"]
    uncached_tokens = reference_tokens["uncached"]
    if cached_tokens is None or uncached_tokens is None:
        raise RuntimeError("benchmark did not produce cached and uncached results")
    cached_vs_uncached = compare_token_ids(cached_tokens, uncached_tokens)
    correctness_failed = not bool(cached_vs_uncached["matched"])
    failure_reason = str(cached_vs_uncached["reason"]) if correctness_failed else None
    if failure_reason is None:
        for mode, result in repeatability.items():
            if not result["matched"]:
                correctness_failed = True
                failure_reason = f"{mode}: {result['reason']}"
                break
    correctness = {
        "matched": not correctness_failed,
        "reason": failure_reason,
        "cached_vs_uncached": cached_vs_uncached,
        "repeatability": repeatability,
    }

    cached_summary = summarize_samples(samples["cached"])
    uncached_summary = summarize_samples(samples["uncached"])
    cached_total = cast(dict[str, float], cached_summary["total_seconds"])
    uncached_total = cast(dict[str, float], uncached_summary["total_seconds"])
    cached_decode = cast(dict[str, float], cached_summary["decode_seconds"])
    uncached_decode = cast(dict[str, float], uncached_summary["decode_seconds"])

    return {
        "prompt_tokens": sequence_length,
        "correctness": correctness,
        "speedup": {
            "total": speedup(uncached_total["median"], cached_total["median"]),
            "decode": speedup(uncached_decode["median"], cached_decode["median"]),
        },
        "cached": {
            "summary": cached_summary,
            "samples": [sample.to_dict() for sample in samples["cached"]],
        },
        "uncached": {
            "summary": uncached_summary,
            "samples": [sample.to_dict() for sample in samples["uncached"]],
        },
    }


def print_summary(cases: list[dict[str, object]]) -> None:
    print("\n  performance summary")
    print("  " + "-" * 76)
    print(
        f"  {'tokens':>8} {'mode':>10} {'median total':>15} "
        f"{'median prefill':>16} {'median decode':>15} "
        f"{'speedup':>10} {'status':>10}"
    )
    print("  " + "-" * 76)
    for case in cases:
        sequence_length = case["prompt_tokens"]
        for mode in ("cached", "uncached"):
            mode_result = cast(dict[str, object], case[mode])
            summary = cast(dict[str, object], mode_result["summary"])
            total = cast(dict[str, float], summary["total_seconds"])
            prefill = cast(dict[str, float], summary["prefill_seconds"])
            decode = cast(dict[str, float], summary["decode_seconds"])
            correctness = cast(dict[str, object], case["correctness"])
            speedups = cast(dict[str, float], case["speedup"])
            status = "OK" if correctness["matched"] else "WARNING"
            speedup_text = (
                f"{speedups['total']:.2f}x" if mode == "cached" else "baseline"
            )
            print(
                f"  {sequence_length:>8} {mode:>10} "
                f"{total['median']:>14.3f}s "
                f"{prefill['median']:>15.3f}s "
                f"{decode['median']:>14.3f}s {speedup_text:>10} "
                f"{status:>10}"
            )
        correctness = cast(dict[str, object], case["correctness"])
        if not correctness["matched"]:
            details = cast(dict[str, object], correctness["cached_vs_uncached"])
            print(f"    warning: {correctness['reason']}")
            if "first_mismatch" in details:
                print(
                    "    first mismatch: "
                    f"position={details['first_mismatch']} "
                    f"cached={details['expected_token']} "
                    f"uncached={details['actual_token']}"
                )
    print("  " + "-" * 76)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark cached and uncached Qwen3 generation"
    )
    parser.add_argument(
        "--prompt",
        default="/no_think Explain paged KV caching in one sentence.",
    )
    parser.add_argument(
        "--seq-lens",
        type=parse_seq_lens,
        default=parse_seq_lens("128,512,1024"),
        help="comma-separated prompt lengths",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-blocks", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--strict-correctness",
        action="store_true",
        help="fail instead of warning when generated outputs differ",
    )
    args = parser.parse_args()

    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be greater than zero")
    if args.runs <= 0:
        parser.error("--runs must be greater than zero")
    if args.warmup < 0:
        parser.error("--warmup must not be negative")
    if args.block_size <= 0:
        parser.error("--block-size must be greater than zero")
    if args.num_blocks <= 0:
        parser.error("--num-blocks must be greater than zero")

    required_blocks = (
        max(args.seq_lens) + args.max_new_tokens + args.block_size - 1
    ) // args.block_size
    if args.num_blocks < required_blocks:
        parser.error(
            f"--num-blocks must be at least {required_blocks} for the largest case"
        )

    torch.manual_seed(args.seed)
    print("=" * 76)
    print("  BareLLM - Cached vs Uncached Generation Benchmark")
    print("=" * 76)
    print(f"\n  model:        {MODEL_ID}")
    print(f"  device:       {DEVICE}")
    print(f"  dtype:        {DTYPE}")
    print(f"  seq_lens:     {args.seq_lens}")
    print(f"  max new:      {args.max_new_tokens}")
    print(f"  runs/warmup:  {args.runs}/{args.warmup}")

    print("\n[1/3] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    base_token_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(DEVICE)
    print(f"  base prompt tokens: {base_token_ids.shape[1]}")

    print("\n[2/3] Loading model and paged cache...")
    _, engine = load_qwen3_engine(
        MODEL_ID,
        use_cache=True,
        max_batch=1,
        block_size=args.block_size,
        num_blocks=args.num_blocks,
        device=DEVICE,
        dtype=DTYPE,
    )
    print("  ready")

    print("\n[3/3] Running benchmark...")
    cases = []
    for sequence_length in args.seq_lens:
        print(f"  measuring prompt length {sequence_length}...")
        token_ids = build_prompt(base_token_ids, sequence_length)
        case = benchmark_case(
            engine,
            token_ids,
            max_new_tokens=args.max_new_tokens,
            runs=args.runs,
            warmup=args.warmup,
            sequence_length=sequence_length,
        )
        cases.append(case)
        correctness = cast(dict[str, object], case["correctness"])
        if args.strict_correctness and not correctness["matched"]:
            raise RuntimeError(
                f"prompt length {sequence_length}: {correctness['reason']}"
            )

    payload = {
        "schema_version": 1,
        "model": MODEL_ID,
        "device": DEVICE,
        "dtype": str(DTYPE),
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "runs": args.runs,
        "warmup": args.warmup,
        "strict_correctness": args.strict_correctness,
        "block_size": args.block_size,
        "num_blocks": args.num_blocks,
        "cases": cases,
    }
    print_summary(cases)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\n  results: {args.output}")


if __name__ == "__main__":
    main()
