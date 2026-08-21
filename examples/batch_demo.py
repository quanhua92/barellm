"""Demonstrate one-token batched decode with independent requests.

Usage:
    uv run python examples/batch_demo.py
    uv run python examples/batch_demo.py "First prompt" "Second prompt"
"""

import argparse
import time

from transformers import AutoTokenizer

from barellm.config import DEVICE, DTYPE, MODEL_ID
from barellm.engine.request import Request
from barellm.runtime import load_qwen3_engine

DEFAULT_PROMPTS = [
    "/no_think Say hello in one short sentence.",
    "/no_think What is 2+2?",
    "/no_think Give three colors.",
]
DEFAULT_LIMITS = [20, 5, 30]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a batched BareLLM demo")
    parser.add_argument("prompts", nargs="*", help="up to three user prompts")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    if len(args.prompts) > len(DEFAULT_PROMPTS):
        parser.error("at most three prompts are supported")

    prompts = args.prompts or DEFAULT_PROMPTS
    limits = DEFAULT_LIMITS[: len(prompts)]

    print("=" * 60)
    print("  BareLLM - Batched Decode Demo")
    print("=" * 60)
    print(f"\n  device: {DEVICE}")
    print(f"  dtype:  {DTYPE}")
    print(f"  batch:  {len(prompts)} requests")

    print("\n[1/3] Loading tokenizer and model...")
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    config, engine = load_qwen3_engine(
        MODEL_ID,
        max_batch=len(prompts),
    )
    print(f"  time: {time.perf_counter() - start:.3f}s")

    eos_ids = {config.eos_token_id}
    if tokenizer.eos_token_id is not None:
        eos_ids.add(tokenizer.eos_token_id)

    requests: list[Request] = []
    print("\n[2/3] Preparing requests...")
    for index, (prompt, max_new_tokens) in enumerate(zip(prompts, limits)):
        messages = [{"role": "user", "content": prompt}]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        token_ids = tokenizer(
            prompt_text,
            return_tensors="pt",
        )["input_ids"].to(DEVICE)

        request_id = f"request-{index + 1}"

        def on_token(token_id: int, _count: int, label=request_id) -> bool:
            piece = tokenizer.decode([token_id], skip_special_tokens=True)
            print(f"\n  [{label}] {piece}", end="", flush=True)
            return True

        request = Request(
            id=request_id,
            token_ids=token_ids,
            max_new_tokens=max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            eos_ids=eos_ids,
            on_token=on_token,
        )
        requests.append(request)
        engine.scheduler.add_request(request)
        print(
            f"  {request_id}: prompt_tokens={request.seq_len}, "
            f"max_new_tokens={max_new_tokens}"
        )

    print("\n[3/3] Decoding shared batches...")
    start = time.perf_counter()
    engine.run()
    elapsed = time.perf_counter() - start

    print("\n\n" + "-" * 60)
    print(f"  elapsed: {elapsed:.3f}s")
    for request in requests:
        print(
            f"  {request.id}: generated={request.generated_count}, "
            f"finish={request.finish_reason}, stop={request.stop_reason}"
        )
    print("-" * 60)


if __name__ == "__main__":
    main()
