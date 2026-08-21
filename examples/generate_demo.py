"""Load Qwen3-0.6B and generate text with BareLLM's public API.

Usage:
    uv run python examples/generate_demo.py
    uv run python examples/generate_demo.py "Say hello world"
    uv run python examples/generate_demo.py --no-cache "Say hello world"
    uv run python examples/generate_demo.py --profile "Say hello world"
"""

import argparse
import time
from pathlib import Path

from transformers import AutoTokenizer

from barellm.config import DEVICE, DTYPE, MODEL_ID
from barellm.engine import TorchProfiler, TraceRecorder, generate, profile_run_dir
from barellm.runtime import load_qwen3_engine


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate text with BareLLM's public generate API"
    )
    parser.add_argument("prompt", nargs="?", default="/no_think What is 2+2?")
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="use full-sequence recomputation as a correctness reference",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="export lightweight engine and metrics profiles",
    )
    parser.add_argument(
        "--torch-profile",
        action="store_true",
        help="also export the large PyTorch operator profile",
    )
    parser.add_argument(
        "--profile-dir",
        type=Path,
        help="profile output directory; enables profiling",
    )
    args = parser.parse_args()

    if args.max_new_tokens < 0:
        parser.error("--max-new-tokens must be non-negative")

    print("=" * 60)
    print("  BareLLM - Qwen3-0.6B Text Generation API Demo")
    print("=" * 60)
    print(f"\n  device:       {DEVICE}")
    print(f"  dtype:        {DTYPE}")
    print(f"  cache:        {'off' if args.no_cache else 'paged'}")
    profiling = args.profile or args.profile_dir is not None or args.torch_profile
    torch_profiling = args.torch_profile
    profile_dir = args.profile_dir
    if profiling and profile_dir is None:
        profile_dir = profile_run_dir(
            model_name=MODEL_ID,
            device=DEVICE,
        )
    print(f"  profiling:    {'on' if profiling else 'off'}")
    if profile_dir is not None:
        print(f"  profile dir:  {profile_dir}")
    print(f"  prompt:       {args.prompt}")

    print("\n[1/4] Loading tokenizer...")
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    print(f"  vocab:        {tokenizer.vocab_size}")
    print(f"  time:         {time.perf_counter() - start:.3f}s")

    print("\n[2/4] Loading model...")
    start = time.perf_counter()
    config, engine = load_qwen3_engine(
        MODEL_ID,
        use_cache=not args.no_cache,
    )
    print(f"  parameters:   {sum(p.numel() for p in engine.model.parameters()):,}")
    print(f"  time:         {time.perf_counter() - start:.3f}s")

    print("\n[3/4] Preparing prompt...")
    messages = [{"role": "user", "content": args.prompt}]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    token_ids = tokenizer(
        prompt_text,
        return_tensors="pt",
    )["input_ids"].to(DEVICE)
    print(f"  prompt tokens: {token_ids.shape[1]}")

    print(f"\n[4/4] Generating (max {args.max_new_tokens} tokens)...")
    print("\n  output: ", end="", flush=True)

    def on_token(token_id: int, _count: int) -> bool:
        piece = tokenizer.decode([token_id], skip_special_tokens=True)
        print(piece, end="", flush=True)
        return True

    eos_ids = {config.eos_token_id}
    if tokenizer.eos_token_id is not None:
        eos_ids.add(tokenizer.eos_token_id)

    recorder = TraceRecorder() if profiling else None

    def run_generation():
        return generate(
            engine,
            token_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            eos_ids=eos_ids,
            on_token=on_token,
            on_event=recorder,
            use_cache=not args.no_cache,
        )

    if torch_profiling:
        assert profile_dir is not None
        with TorchProfiler(profile_dir / "torch.trace.json"):
            result = run_generation()
    else:
        result = run_generation()

    if profiling:
        assert recorder is not None
        assert profile_dir is not None
        metadata = {
            "model": MODEL_ID,
            "device": DEVICE,
            "dtype": str(DTYPE),
            "use_cache": not args.no_cache,
            "prompt": args.prompt,
        }
        recorder.export_chrome_trace(
            profile_dir / "engine.trace.json",
            metadata=metadata,
        )
        recorder.export_metrics(
            profile_dir / "metrics.json",
            result.metrics,
            metadata=metadata,
        )
    metrics = result.metrics

    print("\n\n" + "-" * 60)
    print(f"  generated:    {result.generated_count} tokens")
    print(f"  finish:       {result.finish_reason}")
    print(f"  stop reason:  {result.stop_reason}")
    print(f"  total time:   {metrics.total_seconds:.3f}s")
    print(f"  TTFT:         {metrics.time_to_first_token or 0.0:.3f}s")
    print(f"  avg ITL:      {metrics.average_inter_token_latency or 0.0:.3f}s")
    print(f"  prefill tok/s: {metrics.prefill_tokens_per_second:.1f}")
    print(f"  decode tok/s:  {metrics.decode_tokens_per_second:.1f}")
    print("-" * 60)


if __name__ == "__main__":
    main()
