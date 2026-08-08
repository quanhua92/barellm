import argparse


def serve_command(args):
    print(f"Starting BareLLM server on {args.host}:{args.port}")


def main():
    parser = argparse.ArgumentParser(
        description="BareLLM: A Minimal AI Inference Engine"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Serve
    serve_parser = subparsers.add_parser("serve", help="Start the HTTP API Server")
    serve_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host IP")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port number")

    args = parser.parse_args()

    if args.command == "serve":
        serve_command(args)


if __name__ == "__main__":
    main()
