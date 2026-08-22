"""Benchmark the current dense-gather paged KV-cache backend.

Usage:
    uv run python examples/benchmark_paged_cache.py
    uv run python examples/benchmark_paged_cache.py --seq-lens 128,512 --runs 3
    uv run python examples/benchmark_paged_cache.py --no-boundaries
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch

from barellm.benchmark import summarize
from barellm.config import DEVICE, DTYPE
from barellm.engine.batched_kv_cache import BatchKVCache
from barellm.engine.block_pool import BlockPool
from barellm.engine.kv_cache_manager import KVCacheManager
from barellm.engine.paged_kv_cache import PagedKVCache
from barellm.engine.request import Request
from barellm.models.cache import KVCache


@dataclass
class PreparedRequest:
    request: Request
    cache: KVCache
    keys: list[torch.Tensor]
    values: list[torch.Tensor]


def parse_positive_list(value: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "values must be comma-separated integers"
        ) from exc
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("values must be positive")
    return values


def parse_batch_layouts(value: str) -> list[list[int]]:
    layouts = [parse_positive_list(group) for group in value.split(";")]
    if not layouts or any(not layout for layout in layouts):
        raise argparse.ArgumentTypeError("batch layouts cannot be empty")
    return layouts


def unique(values: list[int]) -> list[int]:
    return list(dict.fromkeys(values))


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


def required_blocks(lengths: list[int], block_size: int, extra: int = 0) -> int:
    return sum((length + extra + block_size - 1) // block_size for length in lengths)


def make_tensor(
    request_index: int,
    layer_index: int,
    length: int,
    *,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    offset: float,
) -> torch.Tensor:
    count = num_kv_heads * length * head_dim
    values = torch.arange(count, dtype=torch.float32, device=device)
    values = (values.remainder(97) / 97.0) + request_index + layer_index
    return (values.reshape(1, num_kv_heads, length, head_dim) + offset).to(dtype)


def prepare_requests(
    lengths: list[int],
    *,
    block_size: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    capacity_extra: int = 0,
) -> tuple[KVCacheManager, list[PreparedRequest]]:
    max_blocks = required_blocks(lengths, block_size, capacity_extra)
    pool = BlockPool(max_blocks)
    storage = PagedKVCache(
        num_layers=num_layers,
        max_blocks=max_blocks,
        num_kv_heads=num_kv_heads,
        block_size=block_size,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
    )
    manager = KVCacheManager(block_size, pool, storage)
    prepared: list[PreparedRequest] = []

    for request_index, length in enumerate(lengths):
        request = Request(
            id=f"cache-benchmark-{request_index}",
            token_ids=torch.zeros(
                (1, length + capacity_extra),
                dtype=torch.long,
            ),
        )
        if not manager.allocate_request(request):
            raise RuntimeError("failed to allocate benchmark cache")
        cache = manager.get_cache(request)
        keys = []
        values = []
        for layer_index in range(num_layers):
            key = make_tensor(
                request_index,
                layer_index,
                length,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                dtype=dtype,
                device=device,
                offset=0.0,
            )
            value = make_tensor(
                request_index,
                layer_index,
                length,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                dtype=dtype,
                device=device,
                offset=0.5,
            )
            cache.layer(layer_index).append(key, value)
            keys.append(key)
            values.append(value)
        prepared.append(PreparedRequest(request, cache, keys, values))

    return manager, prepared


def release_requests(manager: KVCacheManager, requests: list[PreparedRequest]) -> None:
    for prepared in requests:
        manager.free_request(prepared.request.id)


def verify_single(requests: list[PreparedRequest]) -> None:
    for prepared in requests:
        for layer_index, (expected_key, expected_value) in enumerate(
            zip(prepared.keys, prepared.values)
        ):
            actual_key, actual_value = prepared.cache.layer(layer_index).read()
            torch.testing.assert_close(actual_key, expected_key)
            torch.testing.assert_close(actual_value, expected_value)


def verify_batch(requests: list[PreparedRequest]) -> None:
    batch_cache = BatchKVCache([prepared.cache for prepared in requests])
    max_length = max(item.keys[0].shape[2] for item in requests)
    for layer_index in range(len(requests[0].keys)):
        actual_key, actual_value = batch_cache.layer(layer_index).read()
        assert actual_key.shape[2] == max_length
        assert actual_value.shape == actual_key.shape
        for row, prepared in enumerate(requests):
            length = prepared.keys[layer_index].shape[2]
            torch.testing.assert_close(
                actual_key[row : row + 1, :, :length, :],
                prepared.keys[layer_index],
            )
            torch.testing.assert_close(
                actual_value[row : row + 1, :, :length, :],
                prepared.values[layer_index],
            )
            if length < max_length:
                assert torch.count_nonzero(actual_key[row, :, length:, :]) == 0
                assert torch.count_nonzero(actual_value[row, :, length:, :]) == 0


def measure(
    operation,
    *,
    runs: int,
    warmup: int,
    device: torch.device,
) -> dict[str, object]:
    for _ in range(warmup):
        operation()
    synchronize(device)

    samples = []
    for _ in range(runs):
        synchronize(device)
        started_at = time.perf_counter()
        operation()
        synchronize(device)
        samples.append(time.perf_counter() - started_at)
    return {
        "summary": summarize(samples),
        "samples": samples,
    }


def read_single(requests: list[PreparedRequest]) -> None:
    for prepared in requests:
        for layer_index in range(len(prepared.keys)):
            prepared.cache.layer(layer_index).read()


def read_batch(requests: list[PreparedRequest]) -> None:
    batch_cache = BatchKVCache([prepared.cache for prepared in requests])
    for layer_index in range(len(requests[0].keys)):
        batch_cache.layer(layer_index).read()


def append_all(requests: list[PreparedRequest]) -> None:
    for request_index, prepared in enumerate(requests):
        for layer_index in range(len(prepared.keys)):
            key = make_tensor(
                request_index,
                layer_index,
                1,
                num_kv_heads=prepared.keys[layer_index].shape[1],
                head_dim=prepared.keys[layer_index].shape[3],
                dtype=prepared.keys[layer_index].dtype,
                device=prepared.keys[layer_index].device,
                offset=1.0,
            )
            value = key + 0.5
            prepared.cache.layer(layer_index).append(key, value)


def run_case(
    lengths: list[int],
    *,
    block_size: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    runs: int,
    warmup: int,
) -> dict[str, object]:
    manager, requests = prepare_requests(
        lengths,
        block_size=block_size,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
    )
    try:
        if len(requests) == 1:
            verify_single(requests)
            operation = read_single
            operation_name = "single_read_all_layers"
        else:
            verify_batch(requests)
            operation = read_batch
            operation_name = "batch_read_all_layers"
        operations = {
            operation_name: measure(
                lambda: operation(requests),
                runs=runs,
                warmup=warmup,
                device=device,
            )
        }
    finally:
        release_requests(manager, requests)

    append_manager, append_requests = prepare_requests(
        lengths,
        block_size=block_size,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
        capacity_extra=warmup + runs,
    )
    try:
        operations["append_and_dense_read_all_layers"] = measure(
            lambda: append_all(append_requests),
            runs=runs,
            warmup=warmup,
            device=device,
        )
    finally:
        release_requests(append_manager, append_requests)

    return {
        "lengths": lengths,
        "batch_size": len(lengths),
        "max_length": max(lengths),
        "block_size": block_size,
        "operations": operations,
    }


def print_summary(cases: list[dict[str, object]]) -> None:
    print("\n  paged-cache performance summary")
    print("  " + "-" * 88)
    print(
        f"  {'lengths':>18} {'block':>6} {'operation':>38} {'median':>12} {'min':>12}"
    )
    print("  " + "-" * 88)
    for case in cases:
        length_values = cast(list[int], case["lengths"])
        lengths = ",".join(str(length) for length in length_values)
        operations = cast(dict[str, dict[str, object]], case["operations"])
        for operation, result in operations.items():
            summary = cast(dict[str, float], result["summary"])
            print(
                f"  {lengths:>18} {case['block_size']:>6} "
                f"{operation:>38} "
                f"{summary['median']:>11.6f}s "
                f"{summary['min']:>11.6f}s"
            )
    print("  " + "-" * 88)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark dense gathering from the paged KV cache"
    )
    parser.add_argument(
        "--seq-lens",
        type=parse_positive_list,
        default=parse_positive_list("128,512,1024"),
    )
    parser.add_argument(
        "--boundary-lens",
        type=parse_positive_list,
        default=parse_positive_list("15,16,17,31,32,33,511,512,513"),
    )
    parser.add_argument("--no-boundaries", action="store_true")
    parser.add_argument(
        "--batch-layouts",
        type=parse_batch_layouts,
        default=parse_batch_layouts("128,128;128,512;512,1024"),
        help="semicolon-separated comma-separated batch lengths",
    )
    parser.add_argument(
        "--block-sizes",
        type=parse_positive_list,
        default=parse_positive_list("8,16,32"),
    )
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.num_layers <= 0:
        parser.error("--num-layers must be greater than zero")
    if args.num_kv_heads <= 0:
        parser.error("--num-kv-heads must be greater than zero")
    if args.head_dim <= 0:
        parser.error("--head-dim must be greater than zero")
    if args.runs <= 0:
        parser.error("--runs must be greater than zero")
    if args.warmup < 0:
        parser.error("--warmup must not be negative")

    lengths = list(args.seq_lens)
    if not args.no_boundaries:
        lengths.extend(args.boundary_lens)
    lengths = unique(lengths)

    device = torch.device(DEVICE)
    torch.manual_seed(args.seed)
    print("=" * 88)
    print("  BareLLM - Paged KV-Cache Dense-Gather Benchmark")
    print("=" * 88)
    print(f"\n  device:       {device}")
    print(f"  dtype:        {DTYPE}")
    print(f"  seq_lens:     {lengths}")
    print(f"  block_sizes:  {args.block_sizes}")
    print(f"  batch_layouts: {args.batch_layouts}")
    print(f"  layers/heads: {args.num_layers}/{args.num_kv_heads}")
    print(f"  head_dim:     {args.head_dim}")
    print(f"  runs/warmup:  {args.runs}/{args.warmup}")

    cases = []
    for block_size in args.block_sizes:
        for length in lengths:
            cases.append(
                run_case(
                    [length],
                    block_size=block_size,
                    num_layers=args.num_layers,
                    num_kv_heads=args.num_kv_heads,
                    head_dim=args.head_dim,
                    dtype=DTYPE,
                    device=device,
                    runs=args.runs,
                    warmup=args.warmup,
                )
            )
        for batch_lengths in args.batch_layouts:
            cases.append(
                run_case(
                    batch_lengths,
                    block_size=block_size,
                    num_layers=args.num_layers,
                    num_kv_heads=args.num_kv_heads,
                    head_dim=args.head_dim,
                    dtype=DTYPE,
                    device=device,
                    runs=args.runs,
                    warmup=args.warmup,
                )
            )

    payload = {
        "schema_version": 1,
        "device": str(device),
        "dtype": str(DTYPE),
        "seed": args.seed,
        "num_layers": args.num_layers,
        "num_kv_heads": args.num_kv_heads,
        "head_dim": args.head_dim,
        "runs": args.runs,
        "warmup": args.warmup,
        "cases": cases,
    }
    print_summary(cases)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\n  results: {args.output}")


if __name__ == "__main__":
    main()
