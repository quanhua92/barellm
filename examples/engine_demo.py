"""Load Qwen3-0.6B and drive the Engine directly.

Usage:
    uv run python examples/engine_demo.py
    uv run python examples/engine_demo.py "Say hello world"
    uv run python examples/engine_demo.py --no-cache "Say hello world"
    uv run python examples/engine_demo.py --profile "Say hello world"
"""

import argparse
import time
from pathlib import Path

from transformers import AutoTokenizer

from barellm.config import DEVICE, DTYPE, MODEL_ID
from barellm.engine import TorchProfiler, TraceRecorder, profile_run_dir
from barellm.engine.block_pool import BlockPool
from barellm.engine.engine import Engine
from barellm.engine.events import MetricsCollector
from barellm.engine.kv_cache_manager import KVCacheManager
from barellm.engine.paged_kv_cache import PagedKVCache
from barellm.engine.request import Request
from barellm.engine.scheduler import Scheduler
from barellm.models.qwen3 import Qwen3ForCausalLM, load_config
from barellm.models.weights import load_into_model, load_weights


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with BareLLM")
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

    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be greater than zero")

    print("=" * 60)
    print("  BareLLM - Qwen3-0.6B Text Generation")
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
    cfg = load_config(MODEL_ID)
    model = Qwen3ForCausalLM.from_config(cfg)
    model.to(device=DEVICE, dtype=DTYPE)
    weights = load_weights(MODEL_ID, device=DEVICE, dtype=DTYPE)
    load_into_model(model, weights, dtype=DTYPE)
    model.eval()
    print(f"  parameters:   {sum(p.numel() for p in model.parameters()):,}")
    print(f"  time:         {time.perf_counter() - start:.3f}s")

    cache_manager: KVCacheManager | None = None
    if args.no_cache:
        print("\n[3/4] Using uncached reference path...")
    else:
        print("\n[3/4] Creating paged KV cache...")
        block_size = 16
        num_blocks = 256
        pool = BlockPool(num_blocks)
        paged_kv = PagedKVCache(
            num_layers=cfg.num_hidden_layers,
            max_blocks=num_blocks,
            num_kv_heads=cfg.num_key_value_heads,
            block_size=block_size,
            head_dim=cfg.head_dim,
            dtype=DTYPE,
            device=DEVICE,
        )
        cache_manager = KVCacheManager(block_size, pool, paged_kv)
        print(f"  blocks:       {num_blocks}")
        print(f"  block size:   {block_size} tokens")

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
    events = []
    recorder = TraceRecorder() if profiling else None

    def on_token(token_id: int, _count: int) -> bool:
        piece = tokenizer.decode([token_id], skip_special_tokens=True)
        print(piece, end="", flush=True)
        return True

    eos_ids = {cfg.eos_token_id}
    if tokenizer.eos_token_id is not None:
        eos_ids.add(tokenizer.eos_token_id)

    request = Request(
        id="demo",
        token_ids=token_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_ids=eos_ids,
        on_token=on_token,
    )
    scheduler = Scheduler(max_batch=1)
    scheduler.add_request(request)
    engine = Engine(model, scheduler, cache_manager)
    event_sink = recorder if recorder is not None else events.append
    metadata = {
        "model": MODEL_ID,
        "device": DEVICE,
        "dtype": str(DTYPE),
        "use_cache": not args.no_cache,
        "prompt": args.prompt,
    }
    if torch_profiling:
        assert profile_dir is not None
        with TorchProfiler(profile_dir / "torch.trace.json"):
            engine.run(use_cache=not args.no_cache, on_event=event_sink)
    else:
        engine.run(use_cache=not args.no_cache, on_event=event_sink)

    if profiling:
        assert recorder is not None
        assert profile_dir is not None
        events = recorder.events
        recorder.export_chrome_trace(
            profile_dir / "engine.trace.json",
            metadata=metadata,
        )

    metrics_collector = MetricsCollector(request.id)
    for event in events:
        metrics_collector.on_event(event)
    metrics = metrics_collector.build(request.generated_count)
    generated = request.generated_count
    if recorder is not None:
        assert profile_dir is not None
        recorder.export_metrics(
            profile_dir / "metrics.json",
            metrics,
            metadata=metadata,
        )

    print("\n\n" + "-" * 60)
    print(f"  generated:    {generated} tokens")
    print(f"  finish:       {request.finish_reason}")
    print(f"  stop reason:  {request.stop_reason}")
    print(f"  total time:   {metrics.total_seconds:.3f}s")
    print(f"  TTFT:         {metrics.time_to_first_token or 0.0:.3f}s")
    print(f"  avg ITL:      {metrics.average_inter_token_latency or 0.0:.3f}s")
    prefill_tokens_per_second = metrics.prefill_tokens_per_second
    if prefill_tokens_per_second is None:
        prefill_tokens_per_second = 0.0
    print(f"  prefill tok/s: {prefill_tokens_per_second:.1f}")
    print(f"  decode tok/s:  {metrics.decode_tokens_per_second or 0.0:.1f}")
    print("-" * 60)


if __name__ == "__main__":
    main()
