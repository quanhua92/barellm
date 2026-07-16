import argparse
import sys


def add_run_parser(subparsers) -> None:
    run = subparsers.add_parser("run", help="Generate text from a prompt")
    run.add_argument("prompt", help="Prompt to generate text from")
    run.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="Maximum number of new tokens to generate",
    )
    run.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature"
    )
    run.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    run.add_argument("--model", help="Model name to use", default="Qwen/Qwen3-0.6B")
    run.add_argument(
        "--local_files_only",
        action="store_true",
        help="Use local files only instead of downloading from Hugging Face Hub",
    )
    run.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print generation stats (dtype, device, throughput, etc.)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="barellm",
        description="LLM inference engine built from scratch",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("ls", help="List downloaded models")

    pull = subparsers.add_parser("pull", help="Download model weights")
    pull.add_argument("model", help="Model name to download", default="Qwen/Qwen3-0.6B")

    rm = subparsers.add_parser("rm", help="Remove a downloaded model")
    rm.add_argument("model", help="Model name to remove")

    add_run_parser(subparsers)

    subparsers.add_parser("serve", help="Start the HTTP API server")

    return parser


def cmd_pull(args: argparse.Namespace) -> None:
    from barellm.hub import download_model

    download_model(args.model)


def cmd_ls(args: argparse.Namespace) -> None:
    from barellm.hub import list_models

    models = list_models()
    if models:
        print("Downloaded models:")
        for model in models:
            print(f"- {model}")
    else:
        print("No models downloaded.")


def cmd_rm(args: argparse.Namespace) -> None:
    from barellm.hub import remove_model

    if remove_model(args.model):
        print(f"Model '{args.model}' removed successfully.")
    else:
        print(f"Model '{args.model}' not found.")


def cmd_run(args: argparse.Namespace) -> None:
    from barellm.generate import generate

    text, stats = generate(
        args.prompt,
        args.max_new_tokens,
        args.temperature,
        args.top_p,
        args.local_files_only,
        args.verbose,
    )
    print(text)
    if stats:
        print(_format_stats(stats), file=sys.stderr)


def _format_stats(stats: dict) -> str:
    parts = [
        f"dtype={stats['dtype']}",
        f"device={stats['device']}",
        f"attn_impl={stats['attn_impl']}",
    ]
    if "sdpa_backends" in stats:
        b = stats["sdpa_backends"]
        parts.append(
            f"sdpa[flash={b['flash']} mem_eff={b['mem_efficient']} "
            f"math={b['math']} cudnn={b['cudnn']}]"
        )
    parts.append(f"prompt_tokens={stats['prompt_tokens']}")
    parts.append(f"generated_tokens={stats['generated_tokens']}")
    parts.append(f"total_time={stats['total_time']:.2f}s")
    parts.append(f"throughput={stats['throughput']:.1f} tok/s")
    return " ".join(parts)


COMMANDS = {
    "pull": cmd_pull,
    "ls": cmd_ls,
    "rm": cmd_rm,
    "run": cmd_run,
}


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    handler = COMMANDS.get(args.command)
    if handler is not None:
        handler(args)
    else:
        print(f"TODO: `{args.command}` is not implemented yet. See docs/ROADMAP.md")


if __name__ == "__main__":
    main()
