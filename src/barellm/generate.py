import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from barellm.config import DEVICE, DTYPE, MODEL_ID


def _sync() -> None:
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elif DEVICE == "mps":
        torch.mps.synchronize()


def generate(
    prompt: str,
    max_new_tokens: int = 300,
    temperature: float = 0.7,
    top_p: float = 0.9,
    local_files_only: bool = False,
    verbose: bool = False,
) -> tuple[str, dict[str, Any] | None]:
    kwargs: dict[str, Any] = {
        "local_files_only": local_files_only,
    }
    if DEVICE != "mps":
        kwargs["device_map"] = DEVICE

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **kwargs)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE, **kwargs)

    if DEVICE == "mps":
        model = model.to(DEVICE)

    model.eval()

    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    ).to(DEVICE)

    with torch.inference_mode():
        if verbose:
            _sync()
            start = time.perf_counter()
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )
        if verbose:
            _sync()
            total_time = time.perf_counter() - start

    prompt_tokens = inputs["input_ids"].shape[1]
    generated_tokens = output[0].shape[0] - prompt_tokens
    text = tokenizer.decode(output[0][prompt_tokens:], skip_special_tokens=True)

    if not verbose:
        return text, None

    stats: dict[str, Any] = {
        "dtype": str(DTYPE),
        "device": DEVICE,
        "attn_impl": model.config._attn_implementation,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "total_time": total_time,
        "throughput": generated_tokens / total_time if total_time > 0 else 0.0,
    }
    if DEVICE == "cuda":
        stats["sdpa_backends"] = {
            "flash": torch.backends.cuda.flash_sdp_enabled(),
            "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math": torch.backends.cuda.math_sdp_enabled(),
            "cudnn": torch.backends.cuda.cudnn_sdp_enabled(),
        }

    return text, stats
