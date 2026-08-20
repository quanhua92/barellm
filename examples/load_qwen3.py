import argparse
import time
from collections.abc import Callable

import torch

from barellm.config import DEVICE, DTYPE, MODEL_ID
from barellm.models.qwen3 import Qwen3ForCausalLM, load_config
from barellm.models.weights import load_into_model, load_weights


def sync() -> None:
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elif DEVICE == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


def timed[T](label: str, fn: Callable[[], T]) -> tuple[T, float]:
    start = time.perf_counter()
    result = fn()
    sync()
    elapsed = time.perf_counter() - start
    print(f"  {label:<24} {elapsed:>7.3f}s")
    return result, elapsed


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def main() -> None:
    parser = argparse.ArgumentParser(description="Load Qwen3-0.6B into BareLLM")
    parser.add_argument("--seq-len", type=int, default=1024, help="Sequence length")
    parser.add_argument("--runs", type=int, default=5, help="Benchmark runs")
    args = parser.parse_args()
    if args.seq_len <= 0 or args.runs <= 0:
        parser.error("--seq-len and --runs must be greater than zero")

    print("=" * 60)
    print("  BareLLM — Qwen3-0.6B Load & Forward Benchmark")
    print("=" * 60)
    print(f"\n  device:  {DEVICE}")
    print(f"  dtype:   {DTYPE}")
    print(f"  model:   {MODEL_ID}")
    print(f"  seq_len: {args.seq_len}")
    print(f"  runs:    {args.runs}")

    print("\n[1/4] Loading config...")
    cfg, t_cfg = timed("parse config.json", lambda: load_config(MODEL_ID))
    print("\n  architecture:")
    print(f"    layers:     {cfg.num_hidden_layers}")
    print(f"    hidden:     {cfg.hidden_size}")
    print(f"    heads:      {cfg.num_attention_heads}")
    print(f"    kv_heads:   {cfg.num_key_value_heads}")
    print(f"    head_dim:   {cfg.head_dim}")
    print(f"    inter:      {cfg.intermediate_size}")
    print(f"    vocab:      {cfg.vocab_size}")
    print(f"    rope_theta: {cfg.rope_theta}")
    print("    qk_norm:    True")

    print("\n[2/4] Creating model...")
    model, t_create = timed("instantiate", lambda: Qwen3ForCausalLM.from_config(cfg))
    n_params = count_params(model)
    bytes_per = torch.empty((), dtype=DTYPE).element_size()
    size_mb = n_params * bytes_per / 1e6
    print(f"\n  params:       {n_params:,}")
    print(f"  size (dtype): {size_mb:.1f} MB ({bytes_per} bytes/param)")

    print("\n[3/4] Loading weights...")
    model.to(device=DEVICE, dtype=DTYPE)
    weights, t_load_w = timed(
        "read safetensors", lambda: load_weights(MODEL_ID, DEVICE, DTYPE)
    )
    print(f"  tensors:      {len(weights)}")
    _, t_load_m = timed(
        "load_into_model", lambda: load_into_model(model, weights, dtype=DTYPE)
    )
    model.eval()
    print(f"\n  total load:   {t_load_w + t_load_m:.3f}s")

    print(f"\n[4/4] Forward pass (seq_len={args.seq_len}, {args.runs} runs)...")
    token_ids = torch.randint(0, cfg.vocab_size, (1, args.seq_len), device=DEVICE)

    with torch.inference_mode():
        logits = model(token_ids)
    sync()

    timings: list[float] = []
    with torch.inference_mode():
        for i in range(args.runs):
            start = time.perf_counter()
            logits = model(token_ids)
            sync()
            elapsed = time.perf_counter() - start
            timings.append(elapsed)
            print(f"  run {i + 1}:  {elapsed:.3f}s")

    avg = sum(timings) / len(timings)
    best = min(timings)
    tok_per_s = args.seq_len / best

    print(f"\n  {'avg':<24} {avg:>7.3f}s")
    print(f"  {'best':<24} {best:>7.3f}s")
    print(f"  {'prefill tok/s (best)':<24} {tok_per_s:>7.1f}")

    print(f"\n  logits shape:       {tuple(logits.shape)}")
    print(f"  predicted token id: {logits[0, -1].argmax().item()}")
    print(
        f"  logits range:       [{logits.min().item():.2f}, {logits.max().item():.2f}]"
    )
    print(f"  has nan:            {torch.isnan(logits).any().item()}")

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  config load:        {t_cfg:.3f}s")
    print(f"  model create:       {t_create:.3f}s")
    print(f"  weights load:       {t_load_w + t_load_m:.3f}s")
    print(f"  forward (best):     {best:.3f}s  ({tok_per_s:.1f} prefill tok/s)")
    print(f"  total params:       {n_params:,}")
    print(f"  device/dtype:       {DEVICE} / {DTYPE}")


if __name__ == "__main__":
    main()
