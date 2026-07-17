from transformers import AutoConfig

model_ids = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-32B",
]

for model_id in model_ids:
    config = AutoConfig.from_pretrained(model_id)
    print(f"===== {model_id} =====")
    print(config)
    print("-" * 80)

print(
    f"\n{'Model':<18} {'hidden_size':>12} {'mlp_dim':>8} {'layers':>7} {'q_heads':>8} {'kv_heads':>8} {'head_dim':>8} {'vocab':>8} {'max_ctx':>8}"
)
print("-" * 100)

for model_id in model_ids:
    config = AutoConfig.from_pretrained(model_id)
    print(
        f"{model_id.split('/')[1]:<18} "
        f"{config.hidden_size:>12} "
        f"{config.intermediate_size:>8} "
        f"{config.num_hidden_layers:>7} "
        f"{config.num_attention_heads:>8} "
        f"{config.num_key_value_heads:>8} "
        f"{config.head_dim:>8} "
        f"{config.vocab_size:>8} "
        f"{config.max_position_embeddings:>8}"
    )

print("\nbytes per token = 2 * num_layers * num_kv_heads * head_dim * dtype_bytes")
print(f"{'Model':<18} {'bytes/token':>16} {'1K ctx':>8} {'4K ctx':>8} {'32K ctx':>9}")
print("-" * 80)

for model_id in model_ids:
    config = AutoConfig.from_pretrained(model_id)

    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    dtype_bytes = 2  # bfloat16 / float16

    # bytes per token
    bpt = 2 * num_layers * num_kv_heads * head_dim * dtype_bytes
    print(
        f"{model_id:<18} {bpt:>16} "
        f"{bpt * 1024 / 1e6:>6.0f}MB "
        f"{bpt * 1024 * 4 / 1e6:>6.0f}MB "
        f"{bpt * 1024 * 32 / 1e9:>6.0f}GB "
    )

print(
    "\nConcurrent sequences on 24GB GPU (4K context) - ignoring weights and activations"
)
print(f"{'Model':<18}  {'Concurrent':>12}")
print("-" * 80)
for model_id in model_ids:
    config = AutoConfig.from_pretrained(model_id)

    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    dtype_bytes = 2  # bfloat16 / float16

    # bytes per token
    bpt = 2 * num_layers * num_kv_heads * head_dim * dtype_bytes
    available = 24e9  # 24GB, ignoring weights and activations
    per_seq = bpt * 1024 * 4
    max_seq = int(available / per_seq)
    print(f"{model_id:<18}  {max_seq:>12}")
