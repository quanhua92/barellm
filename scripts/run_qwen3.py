import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen3-0.6B"

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

if device == "cuda":
    dtype = torch.bfloat16
elif device == "mps":
    dtype = torch.float16
else:
    dtype = torch.float32

print(f"dtype={dtype} device={device}")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map=device)
model.eval()

print(f"model={model}")
print(f"tokenizer={tokenizer}")

messages = [{"role": "user", "content": "Explain attention layer in one sentence."}]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = tokenizer(prompt, return_tensors="pt").to(device)

print(f"prompt={prompt}")
print(f"inputs={inputs}")

with torch.inference_mode():
    output = model.generate(
        **inputs, max_new_tokens=300, temperature=0.7, top_p=0.9, do_sample=True
    )

response = tokenizer.decode(
    output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
)
print(f"output={output}")
print(f"response={response}")
