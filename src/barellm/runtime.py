import torch

from barellm.attention import AttentionBackendName
from barellm.config import DEVICE, DTYPE, MODEL_ID
from barellm.engine.block_pool import BlockPool
from barellm.engine.engine import Engine
from barellm.engine.kv_cache_manager import KVCacheManager
from barellm.engine.paged_kv_cache import PagedKVCache
from barellm.engine.scheduler import Scheduler
from barellm.models.qwen3 import Qwen3Config, Qwen3ForCausalLM, load_config
from barellm.models.weights import load_into_model, load_weights


def load_qwen3_engine(
    model_id: str = MODEL_ID,
    *,
    use_cache: bool = True,
    max_batch: int = 1,
    block_size: int = 16,
    num_blocks: int = 256,
    device: str = DEVICE,
    dtype: torch.dtype = DTYPE,
    attention_backend: AttentionBackendName = "sdpa",
) -> tuple[Qwen3Config, Engine]:
    """Load a Qwen3 checkpoint and construct a BareLLM engine."""
    config = load_config(model_id)
    model = Qwen3ForCausalLM.from_config(
        config,
        attention_backend=attention_backend,
    )
    model.to(device=device, dtype=dtype)

    weights = load_weights(model_id, device=device, dtype=dtype)
    load_into_model(model, weights, dtype=dtype)
    model.eval()

    cache_manager = None
    if use_cache:
        block_pool = BlockPool(num_blocks)
        paged_kv_cache = PagedKVCache(
            num_layers=config.num_hidden_layers,
            max_blocks=num_blocks,
            num_kv_heads=config.num_key_value_heads,
            block_size=block_size,
            head_dim=config.head_dim,
            dtype=dtype,
            device=device,
        )
        cache_manager = KVCacheManager(
            block_size=block_size,
            block_pool=block_pool,
            paged_kv_cache=paged_kv_cache,
        )

    return config, Engine(
        model=model,
        scheduler=Scheduler(max_batch=max_batch),
        kv_cache_manager=cache_manager,
    )
