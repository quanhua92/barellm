from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from barellm.config import DEVICE, DTYPE, MODEL_ID


def generate(
    prompt: str,
    max_new_tokens: int = 300,
    temperature: float = 0.7,
    top_p: float = 0.9,
    local_files_only: bool = False,
) -> str:
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
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )

    return tokenizer.decode(
        output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
