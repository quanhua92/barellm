import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="barellm",
        description="LLM inference engine built from scratch",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("pull", help="Download model weights")
    subparsers.add_parser("ls", help="List downloaded models")
    rm = subparsers.add_parser("rm", help="Remove a downloaded model")
    rm.add_argument("model", help="Model name to remove")
    subparsers.add_parser("run", help="Generate text from a prompt")
    subparsers.add_parser("serve", help="Start the HTTP API server")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    print(f"TODO: `{args.command}` is not implemented yet. See docs/ROADMAP.md")


if __name__ == "__main__":
    main()
