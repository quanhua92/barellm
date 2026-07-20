import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from barellm.config import DEVICE, DTYPE

model_id = "Qwen/Qwen3-0.6B"

device = DEVICE
dtype = DTYPE
print(f"dtype={dtype} device={device}")
tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, dtype=dtype, local_files_only=True
)
model = model.to(device)
model.eval()

print(f"attn_impl={model.config._attn_implementation}")
if device == "cuda":
    print(
        f"sdpa backends: flash={torch.backends.cuda.flash_sdp_enabled()} "
        f"mem_efficient={torch.backends.cuda.mem_efficient_sdp_enabled()} "
        f"math={torch.backends.cuda.math_sdp_enabled()} "
        f"cudnn={torch.backends.cuda.cudnn_sdp_enabled()}"
    )

messages = [{"role": "user", "content": "What is the capital of Vietnam?"}]
inputs = tokenizer.apply_chat_template(
    messages, return_tensors="pt", add_generation_prompt=True, enable_thinking=False
).to(DEVICE)


print(f"    inputs: {inputs}")
print("\n=== model.forward() ===")
with torch.inference_mode():
    out = model(inputs.input_ids)

print(f"    inputs: {inputs.input_ids.shape}")
print(f"    logits: {out.logits.shape}")
print("    -> forward() returns logits for ALL positions")
print("    -> we only use the LAST position: logits[:, -1, :]")

next_token_id = out.logits[0, -1, :].argmax().item()
print(
    f"    Next token (greedy): {next_token_id} = {tokenizer.decode([next_token_id])!r}"
)

print("\n=== model.generate() ===")
with torch.inference_mode():
    gen_output = model.generate(**inputs, max_new_tokens=10, do_sample=False)

gen_tokens = gen_output[0][inputs.input_ids.shape[1] :]
gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
print(f"    Output tokens: {gen_tokens.tolist()}")
print(f"    Text: {gen_text!r}")

print("\n===Manual loop using forward() ===")
print("   (this is what generate() does internally")

with torch.inference_mode():
    out = model(inputs.input_ids, use_cache=True)
    past_kv = out.past_key_values
    next_token = out.logits[:, -1:].argmax(dim=-1)
    generated = [next_token.item()]

    for step in range(9):
        out = model(next_token, past_key_values=past_kv, use_cache=True)
        past_kv = out.past_key_values
        next_token = out.logits[:, -1:].argmax(dim=-1)
        generated.append(next_token.item())
    manual_text = tokenizer.decode(generated, skip_special_tokens=True)
    print(f"    Output tokens: {generated}")
    print(f"    Text: {manual_text!r}")

print("\n=== Comparison ===")
print(f"    manual == generate(): {manual_text == gen_text}")
