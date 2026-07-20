import torch
from transformers import AutoModelForCausalLM

from barellm.config import DEVICE, DTYPE

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=DTYPE)
model = model.to(DEVICE)
model.eval()

block = model.model.layers[0]
config = model.config

print("\n===ATTENTION CONFIG===")
print(f"    hidden_size: {config.hidden_size}")
print(f"    num_heads: {config.num_attention_heads}")
print(f"    num_kv_heads: {config.num_key_value_heads}")
print(f"    head_dim: {config.head_dim}")

print("\n=== Qwen3 Transformer Block (layer 0)===")
print("Block", block)

print("\n=== Generate input 4096 context===")
B, L, D = 1, 4096, config.hidden_size

x = torch.randn(B, L, D, dtype=DTYPE)
x = x.to(DEVICE)
print(f"Input shape: {x.shape} [B, L, hidden_size]")

print("\n=== Attention branch (pre-norm + residual)===")
residual = x

print(f"    residual = x (saved for bypass) shape: {residual.shape}")

h = block.input_layernorm(x)
print(f"    input_layernorm (RMSNorm): {x.shape} -> {h.shape}")

position_ids = torch.arange(L, device=DEVICE).unsqueeze(0)
position_embeddings = model.model.rotary_emb(h, position_ids)
print(
    f"    position_embeddings shape: cos: {position_embeddings[0].shape} sin: {position_embeddings[1].shape}"
)
h = block.self_attn(h, position_embeddings=position_embeddings, attention_mask=None)[0]
print(f"    self_attn (GQA + QK-Norm + RoPE): -> {h.shape}")

h = residual + h
print(f"    residual + attn_out: -> {h.shape}")

print("\n=== MPL branch (pre-norm + residual) ===")
print("\n=== SwiGLU Internals ===")
print(f"    gate_proj.weight: {block.mlp.gate_proj.weight.shape}")
print(f"    up_proj.weight:   {block.mlp.up_proj.weight.shape}")
print(f"    down_proj.weight: {block.mlp.down_proj.weight.shape}")


residual2 = h
print(f"    residual2 = h (saved for bypass) shape: {residual2.shape}")

h = block.post_attention_layernorm(h)
print(f"    post_attention_layernorm (RMSNorm) -> {h.shape}")

gate = block.mlp.gate_proj(h)
up = block.mlp.up_proj(h)
print(f"    gate_proj: -> {gate.shape} (hidden -> {config.intermediate_size})")
print(f"    up_proj: -> {up.shape} (hidden -> {config.intermediate_size})")

activated = torch.nn.functional.silu(gate) * up
print(f"    SiLU(gate) * up: -> {activated.shape}")

down = block.mlp.down_proj(activated)
print(f"    down_proj: -> {down.shape} (hidden -> {config.intermediate_size})")

h = residual2 + down
print(f"    residual2 + down: -> {h.shape}")

print("\n=== Verification with transformers")
with torch.inference_mode():
    expected = block(x, position_embeddings=position_embeddings, attention_mask=None)[0]

print(f"Manual trace: \n{h}")
print(f"Expected : \n{expected}")

match = torch.allclose(h, expected, atol=1e-2)
print(f"Manual trace vs block.forward(): allclose={match}")
