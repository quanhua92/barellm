"""Load Qwen3-0.6B and generate text with BareLLM's public API.

Usage:
    uv run python examples/generate_demo.py
    uv run python examples/generate_demo.py "Say hello world"
    uv run python examples/generate_demo.py --no-cache "Say hello world"
"""

import argparse
import time

import torch
from transformers import AutoTokenizer

from barellm.config import DEVICE, DTYPE, MODEL_ID
from barellm.engine import generate
from barellm.engine.block_pool import BlockPool
from barellm.engine.engine import Engine
from barellm.engine.kv_cache_manager import KVCacheManager
from barellm.engine.paged_kv_cache import PagedKVCache
from barellm.engine.scheduler import Scheduler
from barellm.models.qwen3 import Qwen3ForCausalLM, load_config
from barellm.models.weights import load_into_model, load_weights


def sync() -> None:
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elif DEVICE == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


def build_engine(config, use_cache: bool) -> Engine:
    model = Qwen3ForCausalLM.from_config(config)
    model.to(device=DEVICE, dtype=DTYPE)

    weights = load_weights(MODEL_ID, device=DEVICE, dtype=DTYPE)
    load_into_model(model, weights, dtype=DTYPE)
    model.eval()

    cache_manager = None
    if use_cache:
        block_size = 16
        num_blocks = 256
        block_pool = BlockPool(num_blocks)
        paged_kv_cache = PagedKVCache(
            num_layers=config.num_hidden_layers,
            max_blocks=num_blocks,
            num_kv_heads=config.num_key_value_heads,
            block_size=block_size,
            head_dim=config.head_dim,
            dtype=DTYPE,
            device=DEVICE,
        )
        cache_manager = KVCacheManager(
            block_size=block_size,
            block_pool=block_pool,
            paged_kv_cache=paged_kv_cache,
        )

    return Engine(
        model=model,
        scheduler=Scheduler(max_batch=1),
        kv_cache_manager=cache_manager,
    )


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
    args = parser.parse_args()

    if args.max_new_tokens < 0:
        parser.error("--max-new-tokens must be non-negative")

    print("=" * 60)
    print("  BareLLM - Qwen3-0.6B Text Generation API Demo")
    print("=" * 60)
    print(f"\n  device:       {DEVICE}")
    print(f"  dtype:        {DTYPE}")
    print(f"  cache:        {'off' if args.no_cache else 'paged'}")
    print(f"  prompt:       {args.prompt}")

    print("\n[1/4] Loading tokenizer...")
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    print(f"  vocab:        {tokenizer.vocab_size}")
    print(f"  time:         {time.perf_counter() - start:.3f}s")

    print("\n[2/4] Loading model...")
    start = time.perf_counter()
    config = load_config(MODEL_ID)
    engine = build_engine(config, use_cache=not args.no_cache)
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

    token_times: list[float] = []
    start = time.perf_counter()

    def on_token(token_id: int, _count: int) -> bool:
        token_times.append(time.perf_counter())
        piece = tokenizer.decode([token_id], skip_special_tokens=True)
        print(piece, end="", flush=True)
        return True

    eos_ids = {config.eos_token_id}
    if tokenizer.eos_token_id is not None:
        eos_ids.add(tokenizer.eos_token_id)

    result = generate(
        engine,
        token_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_ids=eos_ids,
        on_token=on_token,
        use_cache=not args.no_cache,
    )
    sync()

    elapsed = time.perf_counter() - start
    ttft = token_times[0] - start if token_times else 0.0
    if len(token_times) > 1:
        intervals = [
            token_times[i + 1] - token_times[i] for i in range(len(token_times) - 1)
        ]
        avg_itl = sum(intervals) / len(intervals)
    else:
        avg_itl = 0.0
    tok_per_s = 1.0 / avg_itl if avg_itl > 0 else 0.0

    print("\n\n" + "-" * 60)
    print(f"  generated:    {result.generated_count} tokens")
    print(f"  finish:       {result.finish_reason}")
    print(f"  stop reason:  {result.stop_reason}")
    print(f"  total time:   {elapsed:.3f}s")
    print(f"  TTFT:         {ttft:.3f}s")
    print(f"  avg ITL:      {avg_itl:.3f}s")
    print(f"  decode tok/s: {tok_per_s:.1f}")
    print("-" * 60)


if __name__ == "__main__":
    main()
