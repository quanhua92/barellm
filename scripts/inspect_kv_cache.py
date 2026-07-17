import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from barellm.config import DEVICE, DTYPE

model_id = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, dtype=DTYPE, local_files_only=True
)
model = model.to(DEVICE)
model.eval()

print(model)

messages = [{"role": "user", "content": "who are you?"}]
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,  # disable thinking
)
# prompt += "<think>\nI need to answer directly.\n</think>\n"
inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

print("======Prompt======")
print(prompt)
print("======inputs======")
print(inputs)
print("input_ids shape:", inputs["input_ids"].shape)

generated_ids = []

with torch.inference_mode():
    out = model(**inputs, use_cache=True)
    past_kv = out.past_key_values
    next_token = out.logits[:, -1:].argmax(dim=-1)
    generated_ids.append(next_token.item())

print("out.logits shape:", out.logits.shape)
print("======past_kv======")
for i, kv in enumerate(past_kv):
    if i < 2 or i >= len(past_kv) - 2:  # Print first 2 and last 2 layers
        print(f"Layer {i}:")
        print("  Key shape:", kv[0].shape)
        print("  Value shape:", kv[1].shape)
print("  -> [batch, n_kv_heads, seq_len, head_dim]")

with torch.inference_mode():
    for i in range(10):
        out = model(next_token, use_cache=True, past_key_values=past_kv)
        past_kv = out.past_key_values
        next_token = out.logits[:, -1:].argmax(dim=-1)
        generated_ids.append(next_token.item())
        print(f"Step {i + 1}: next_token={next_token.item()}")
        print("  out.logits.shape", out.logits.shape)

        key_new, value_new = past_kv.layers[0].keys, past_kv.layers[0].values
        print("  New Key shape:", key_new.shape)
        print("  New Value shape:", value_new.shape)


text = tokenizer.decode(generated_ids, skip_special_tokens=False)
print("Output:\n", text)
