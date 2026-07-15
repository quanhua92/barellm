from collections import defaultdict

from transformers import AutoModelForCausalLM

model_id = "Qwen/Qwen3-0.6B"
model = AutoModelForCausalLM.from_pretrained(model_id)
model.eval()

cfg = model.config
total = sum(p.numel() for p in model.parameters())

print(f"model: {model_id}")
print(f"total params: {total:,}")
print(f"hidden size: {cfg.hidden_size}")
print(f"num layers: {cfg.num_hidden_layers}")
print(f"num heads: {cfg.num_attention_heads}")
print(f"num kv heads: {cfg.num_key_value_heads}")
print(f"head dim: {cfg.hidden_size // cfg.num_attention_heads}")
print(f"vocab size: {cfg.vocab_size}")
print(f"intermediate size: {cfg.intermediate_size}")

print("\nby category:")
groups = defaultdict(int)
for name, p in model.named_parameters():
    if "embed" in name:
        category = "embedding"
    elif "layers" in name:
        if any(k in name for k in ["q_proj", "k_proj", "v_proj", "o_proj"]):
            category = "layer_attention"
        elif any(k in name for k in ["gate_proj", "up_proj", "down_proj"]):
            category = "layer_mlp"
        elif "norm" in name:
            category = "layer_norm"
        else:
            category = "layer_other"
    elif "norm" in name:
        category = "final_norm"
    else:
        category = "other"
    groups[category] += p.numel()

for key, count in sorted(groups.items()):
    pct = 100 * count / total
    print(f"  {key:30s} {count:>12,}  {pct:5.1f}%")

total = sum(p.numel() for p in model.parameters())
print("-" * 55)
print(f"  {'TOTAL':30s} {total:>12,}")
print(f"\n  Memory (bf16): {total * 2 / 1e9:.2f} GB")

print("\ndecoder layer 0 breakdown:")
for name, p in model.named_parameters():
    if name.startswith("model.layers.0."):
        short = name.split("model.layers.0.", 1)[1]
        print(f"  {short:35s} {str(tuple(p.shape)):20s} {p.numel():>10,}")
