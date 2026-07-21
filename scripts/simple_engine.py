import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from barellm.config import DEVICE, DTYPE, MODEL_ID
from barellm.sampling.sampler import sample
from barellm.sampling.stops import check_stop


class SimpleInferenceEngine:
    def __init__(self, model_id=MODEL_ID) -> None:
        kwargs = {"local_files_only": True}
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=DTYPE, **kwargs
        )
        self.model = self.model.to(DEVICE)  # type: ignore
        self.model.eval()
        self.eos_ids = {
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
        }

    def __repr__(self) -> str:
        tokenizer_name = self.tokenizer.__class__.__name__
        eos_list = sorted(list(self.eos_ids))

        return (
            f"SimpleInferenceEngine("
            f"model='{self.model.config._name_or_path}', "
            f"device='{self.model.device}', "
            f"tokenizer={tokenizer_name}(vocab_size={len(self.tokenizer)}), "
            f"eos_ids={eos_list}"
            f")"
        )

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        enable_thinking: bool = True,
        temperature=0.7,
        top_k: int = 20,
        top_p: float = 0.9,
        stop_strings: list[str] | None = None,
        deadline: float | None = None,
    ):
        """Returns (text, finish_reason, stop_reason)"""
        # 1. Chat template
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        ).to(DEVICE)

        # 2. Prefill
        out = self.model(inputs.input_ids, use_cache=True)
        past_kv = out.past_key_values
        next_token = out.logits[:, -1, :].squeeze(0)

        generated_ids = []
        deadline = deadline or (time.monotonic() + 60)

        # 3. Decode
        for _ in range(max_new_tokens):
            token = sample(next_token, temperature, top_k, top_p)
            token_id = int(token.item())
            generated_ids.append(token_id)

            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

            result = check_stop(
                token_id=token_id,
                max_new_tokens=max_new_tokens,
                eos_ids=self.eos_ids,
                generated_count=len(generated_ids),
                text_so_far=text,
                stop_strings=stop_strings,
                deadline=deadline,
            )

            if result is not None:
                finish_reason, stop_reason = result
                break

            # Forward
            next_input = token.view(1, 1)
            out = self.model(next_input, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_token = out.logits[:, -1, :].squeeze(0)
        else:
            finish_reason = "length"
            stop_reason = None

        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return text, finish_reason, stop_reason


if __name__ == "__main__":
    engine = SimpleInferenceEngine()
    print("Engine:", engine)

    text, reason, stop = engine.generate("Explain FlashAttention in one sentence")
    print(f"Response: {text}")
    print(f"Finish reason: {reason}")
    print(f"Stop Reason: {stop}")
