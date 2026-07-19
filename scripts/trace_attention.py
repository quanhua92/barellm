import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from barellm.config import DEVICE, DTYPE

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=DTYPE)
model = model.to(DEVICE)
model.eval()

layer0 = model.model.layers[0].self_attn
config = model.config

print("\n=== Attention module (layer 0) ===")
print("layer0", layer0)
print("Note: q_norm, k_norm and RoPE are skipped in this trace.")
print("Shapes are correct, values won't match HF exactly")

print("\n===ATTENTION CONFIG===")
print(f"    hidden_size: {config.hidden_size}")
print(f"    num_heads: {config.num_attention_heads}")
print(f"    num_kv_heads: {config.num_key_value_heads}")
print(f"    head_dim: {config.head_dim}")

print("\n=== Generate input 4096 context===")
B, L, D = 1, 4096, config.hidden_size

x = torch.randn(B, L, D, dtype=DTYPE)
x = x.to(DEVICE)
print(f"Input shape: {x.shape} [B, L, hidden_size]")

print("\n===QKV Projections===")
q = layer0.q_proj(x)
k = layer0.k_proj(x)
v = layer0.v_proj(x)

print(f"After layer0.q_proj shape: {q.shape} (16 heads x 128 = 2048)")
print(f"After layer0.k_proj shape: {k.shape} (8 KV heads x 128 = 1024)")
print(f"After layer0.v_proj shape: {v.shape} (8 heads x 128 = 1024)")

print("\n===Reshape to heads: [B, heads, L, dim]===")
q = q.view(B, L, config.num_attention_heads, config.head_dim).transpose(1, 2)
k = k.view(B, L, config.num_key_value_heads, config.head_dim).transpose(1, 2)
v = v.view(B, L, config.num_key_value_heads, config.head_dim).transpose(1, 2)
print(f"Q reshaped: {q.shape} [B, 16, L, 128]")
print(f"K reshaped: {k.shape} [B, 8, L, 128]")
print(f"V reshaped: {v.shape} [B, 8, L, 128]")

print("\n===GQA: expand KV heads: [B, 16, L, dim]===")
print(
    "Interleave [0, 0, 1, 1, 2, 2, ...] to keep adjacent query heads shares the same underlying KV head"
)
n_rep = config.num_attention_heads // config.num_key_value_heads  # 2
k_expanded = k.repeat_interleave(n_rep, dim=1)
v_expanded = v.repeat_interleave(n_rep, dim=1)
print(f"K expanded: {k_expanded.shape} [B, 16, L, 128]")
print(f"V expanded: {v_expanded.shape} [B, 16, L, 128]")

print(
    "\n===Attention scores: [B, 16, L, L] matrix is the O(L^2) cost. B=1, L=4096: 1 * 16 * 4096 * 4096 * 2 bytes BF16 = 0.5 GB per layer ==="
)
scores = torch.matmul(q, k_expanded.transpose(-2, -1)) / (config.head_dim**0.5)
print(f"QK^T / sqrt(d): {scores.shape} [B, 16, L, L]")

print("\n===Causal mask & softmax ===")
mask = torch.tril(torch.ones(L, L)).to(DEVICE)
scores = scores.masked_fill(mask == 0, float("-inf"))
attn_weights = F.softmax(scores, dim=-1)
print(f"After softmax: {attn_weights.shape} [B, 16, L, L]")

print("\n===Aggregate values===")
out = torch.matmul(attn_weights, v_expanded)
print(f"Attn x V: {out.shape} [B, 16, L, 128]")

print("\n===Merge heads and output projection===")
out = out.transpose(1, 2).contiguous().view(B, L, -1)
print(f"After merge: {out.shape} [B, L, 2048]")

out = layer0.o_proj(out)
print(f"After o_proj: {out.shape} [B, L, 1024] (back to hidden size)")

print("\n=== Attention complete ===")
print(
    "For full block trace (RMSNorm -> attention -> residual -> RMSNorm -> SwiGLU -> residual)"
)
print("see scripts/trace_block.py")
