"""Profile Qwen3-0.6B with BareLLM engine and PyTorch traces.

Usage:
    uv run python examples/profile_demo.py "Say hello world"
    uv run python examples/profile_demo.py --profile-dir profiles/debug "Say hello"
"""

import argparse
from pathlib import Path

from transformers import AutoTokenizer

from barellm.config import DEVICE, DTYPE, MODEL_ID
from barellm.engine import (
    TorchProfiler,
    TraceRecorder,
    generate,
    profile_run_dir,
)
from barellm.runtime import load_qwen3_engine


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile Qwen3-0.6B with BareLLM")
    parser.add_argument("prompt", nargs="?", default="/no_think What is 2+2?")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--profile-dir",
        type=Path,
        help="profile output directory; defaults to a timestamped run directory",
    )
    args = parser.parse_args()

    if args.max_new_tokens < 0:
        parser.error("--max-new-tokens must be non-negative")

    print(f"loading {MODEL_ID} on {DEVICE} with {DTYPE}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    config, engine = load_qwen3_engine(
        MODEL_ID,
        use_cache=not args.no_cache,
    )

    messages = [{"role": "user", "content": args.prompt}]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    token_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(DEVICE)
    eos_ids = {config.eos_token_id}
    if tokenizer.eos_token_id is not None:
        eos_ids.add(tokenizer.eos_token_id)

    print(f"prompt tokens: {token_ids.shape[1]}")
    print("output: ", end="", flush=True)

    def on_token(token_id: int, _count: int) -> bool:
        print(
            tokenizer.decode([token_id], skip_special_tokens=True),
            end="",
            flush=True,
        )
        return True

    recorder = TraceRecorder()
    profile_dir = args.profile_dir or profile_run_dir(
        model_name=MODEL_ID,
        device=DEVICE,
    )
    torch_trace_path = profile_dir / "torch.trace.json"
    engine_trace_path = profile_dir / "engine.trace.json"
    metrics_path = profile_dir / "metrics.json"
    metadata = {
        "model": MODEL_ID,
        "device": DEVICE,
        "dtype": str(DTYPE),
        "use_cache": not args.no_cache,
        "prompt": args.prompt,
    }

    with TorchProfiler(torch_trace_path):
        result = generate(
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

    recorder.export_chrome_trace(engine_trace_path, metadata=metadata)
    recorder.export_metrics(metrics_path, result.metrics, metadata=metadata)

    print("\n\nprofile exports:")
    print(f"  engine trace: {engine_trace_path}")
    print(f"  torch trace:  {torch_trace_path}")
    print(f"  metrics:      {metrics_path}")
    print(f"  finish:       {result.finish_reason}")
    print(f"  TTFT:         {result.metrics.time_to_first_token or 0.0:.3f}s")
    print(f"  decode tok/s: {result.metrics.decode_tokens_per_second:.1f}")


if __name__ == "__main__":
    main()
