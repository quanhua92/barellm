from transformers import AutoConfig

model_ids = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-32B",
]

GPU_MEMORY_GB = 24
OVERHEAD_GB = 1  # CUDA workspace + activations
DTYPE_BYTES = 2  # bf16/fp16


def estimate_weight_bytes(config) -> int:
    hidden = config.hidden_size
    inter = config.intermediate_size
    layers = config.num_hidden_layers
    q_dim = config.num_attention_heads * config.head_dim
    kv_dim = config.num_key_value_heads * config.head_dim
    tied = getattr(config, "tie_word_embeddings", False)

    embed = config.vocab_size * hidden
    lm_head = 0 if tied else config.vocab_size * hidden

    per_layer = (
        hidden * q_dim  # q_proj:  hidden -> q_heads * head_dim
        + hidden * kv_dim  # k_proj:  hidden -> kv_heads * head_dim
        + hidden * kv_dim  # v_proj:  hidden -> kv_heads * head_dim
        + q_dim * hidden  # o_proj:  q_heads * head_dim -> hidden
        + config.head_dim  # q_norm
        + config.head_dim  # k_norm
        + hidden * inter  # gate_proj: hidden -> intermediate
        + hidden * inter  # up_proj:   hidden -> intermediate
        + inter * hidden  # down_proj: intermediate -> hidden
        + 2 * hidden  # input_layernorm + post_attention_layernorm
    )

    total_params = embed + lm_head + layers * per_layer + hidden
    return total_params * DTYPE_BYTES


for model_id in model_ids:
    config = AutoConfig.from_pretrained(model_id)
    print(f"===== {model_id} =====")
    print(config)
    print("-" * 80)

print(
    f"\n{'Model':<18} {'hidden':>7} {'mlp_dim':>8} {'layers':>7} {'q_heads':>8} {'kv_heads':>8} {'head_dim':>8} {'vocab':>8} {'max_ctx':>8}"
)
print("-" * 90)

for model_id in model_ids:
    config = AutoConfig.from_pretrained(model_id)
    print(
        f"{model_id.split('/')[1]:<18} "
        f"{config.hidden_size:>7} "
        f"{config.intermediate_size:>8} "
        f"{config.num_hidden_layers:>7} "
        f"{config.num_attention_heads:>8} "
        f"{config.num_key_value_heads:>8} "
        f"{config.head_dim:>8} "
        f"{config.vocab_size:>8} "
        f"{config.max_position_embeddings:>8}"
    )

print("\nbytes per token = 2 * num_layers * num_kv_heads * head_dim * dtype_bytes")
print(f"{'Model':<18} {'bytes/token':>12} {'1K ctx':>8} {'4K ctx':>8} {'32K ctx':>9}")
print("-" * 60)

for model_id in model_ids:
    config = AutoConfig.from_pretrained(model_id)
    bpt = (
        2
        * config.num_hidden_layers
        * config.num_key_value_heads
        * config.head_dim
        * DTYPE_BYTES
    )
    print(
        f"{model_id.split('/')[1]:<18} {bpt:>12} "
        f"{bpt * 1024 / 1e6:>6.0f}MB "
        f"{bpt * 4096 / 1e6:>6.0f}MB "
        f"{bpt * 32768 / 1e9:>7.1f}GB "
    )

print(
    f"\nConcurrent on {GPU_MEMORY_GB}GB GPU (4K context, bf16 weights + {OVERHEAD_GB}GB overhead)"
)
print(
    f"{'Model':<18} {'estimated_weights':>17} {'available':>10} {'per seq':>8} {'concurrent':>11}"
)
print("-" * 60)

for model_id in model_ids:
    config = AutoConfig.from_pretrained(model_id)
    bpt = (
        2
        * config.num_hidden_layers
        * config.num_key_value_heads
        * config.head_dim
        * DTYPE_BYTES
    )
    weight_gb = estimate_weight_bytes(config) / 1e9
    available_gb = GPU_MEMORY_GB - weight_gb - OVERHEAD_GB
    per_seq_mb = bpt * 4096 / 1e6

    if available_gb <= 0:
        print(
            f"{model_id.split('/')[1]:<18} {weight_gb:>15.1f}GB {'N/A':>10} {per_seq_mb:>6.0f}MB {'0':>11}"
        )
    else:
        max_seq = int(available_gb * 1e9 / (bpt * 4096))
        print(
            f"{model_id.split('/')[1]:<18} {weight_gb:>15.1f}GB "
            f"{available_gb:>8.1f}GB {per_seq_mb:>6.0f}MB {max_seq:>11}"
        )
