import argparse
from pathlib import Path

from transformers import AutoTokenizer

from barellm.config import DEVICE, DTYPE, MODEL_ID
from barellm.engine import TorchProfiler, TraceRecorder, generate, profile_run_dir
from barellm.runtime import load_qwen3_engine


def _generate(args: argparse.Namespace) -> None:
    print(f"loading {args.model} on {DEVICE} with {DTYPE}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    config, engine = load_qwen3_engine(
        args.model,
        use_cache=not args.no_cache,
    )
    profiling = args.profile or args.profile_dir is not None or args.torch_profile
    torch_profiling = args.torch_profile
    profile_dir = args.profile_dir
    if profiling and profile_dir is None:
        profile_dir = profile_run_dir(
            model_name=args.model,
            device=DEVICE,
        )
    print(f"profiling: {'on' if profiling else 'off'}")
    if profile_dir is not None:
        print(f"profile dir: {profile_dir}")

    messages = [{"role": "user", "content": args.prompt}]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    token_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"]
    token_ids = token_ids.to(DEVICE)

    eos_ids = {config.eos_token_id}
    if tokenizer.eos_token_id is not None:
        eos_ids.add(tokenizer.eos_token_id)

    def on_token(token_id: int, _count: int) -> bool:
        print(
            tokenizer.decode([token_id], skip_special_tokens=True), end="", flush=True
        )
        return True

    recorder = TraceRecorder() if profiling else None

    def run_generation():
        return generate(
            engine,
            token_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
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
            "model": args.model,
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
    print(f"  generated: {result.generated_count} tokens")
    print(f"  finish:    {result.finish_reason}")
    print(f"  stop:      {result.stop_reason}")
    print(f"  elapsed:   {metrics.total_seconds:.3f}s")
    print(f"  TTFT:      {metrics.time_to_first_token or 0.0:.3f}s")
    print(f"  prefill:   {metrics.prefill_tokens_per_second:.1f} tok/s")
    print(f"  decode:    {metrics.decode_tokens_per_second:.1f} tok/s")
    print("-" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="BareLLM command-line tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser(
        "generate",
        help="generate text with a Qwen3-compatible model",
    )
    generate_parser.add_argument("--model", default=MODEL_ID)
    generate_parser.add_argument("--prompt", required=True)
    generate_parser.add_argument("--max-new-tokens", type=int, default=128)
    generate_parser.add_argument("--temperature", type=float, default=0.7)
    generate_parser.add_argument("--top-k", type=int, default=0)
    generate_parser.add_argument("--top-p", type=float, default=0.9)
    generate_parser.add_argument("--no-cache", action="store_true")
    generate_parser.add_argument(
        "--profile",
        action="store_true",
        help="export lightweight engine and metrics profiles",
    )
    generate_parser.add_argument(
        "--torch-profile",
        action="store_true",
        help="also export the large PyTorch operator profile",
    )
    generate_parser.add_argument(
        "--profile-dir",
        type=Path,
        help="profile output directory; enables profiling",
    )
    subparsers.add_parser(
        "serve",
        help="start the HTTP server using BARELLM_* environment settings",
    )

    args = parser.parse_args()
    if args.command == "generate":
        if args.max_new_tokens < 0:
            generate_parser.error("--max-new-tokens must be non-negative")
        if args.temperature < 0:
            generate_parser.error("--temperature must be non-negative")
        if args.top_k < 0:
            generate_parser.error("--top-k must be non-negative")
        if not 0.0 < args.top_p <= 1.0:
            generate_parser.error("--top-p must be in the range (0, 1]")
        _generate(args)
    elif args.command == "serve":
        from barellm.web import run_server

        run_server()
