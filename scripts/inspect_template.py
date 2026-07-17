from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

print("======Tokenizer======")
print(tokenizer)

print("======Tokens======")
print(f"eos_token: {tokenizer.eos_token!r} (id={tokenizer.eos_token_id})")
print(f"bos_token: {tokenizer.bos_token!r} (id={tokenizer.bos_token_id})")
print(f"pad_token: {tokenizer.pad_token!r} (id={tokenizer.pad_token_id})")


messages = [{"role": "user", "content": "Hello, how are you?"}]
formatted = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

print("======Repr======")
print(repr(formatted))
print("======Readable======")
print(formatted)

ids = tokenizer(formatted, add_special_tokens=True).input_ids
print("======Token IDs======")
print(ids)

decoded = tokenizer.decode(ids, skip_special_tokens=False)
print("======Decoded Repr======")
print(repr(decoded))
print("======Decoded Readable======")
print(decoded)
print(f"Roundtrip match: {formatted == decoded}")
