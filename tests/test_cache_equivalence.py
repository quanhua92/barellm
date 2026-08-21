import pytest
import torch

from barellm.engine.contiguous_kv_cache import ContiguousKVCache
from barellm.models.qwen3 import Qwen3ForCausalLM


@pytest.mark.parametrize(
    "num_kv_heads",
    [4, 2, 1],
    ids=["mha", "gqa", "mqa"],
)
def test_contiguous_cached_decode_matches_full_forward(
    num_kv_heads: int,
) -> None:
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        head_dim=4,
        num_layers=2,
        num_heads=4,
        num_kv_heads=num_kv_heads,
        use_qk_norm=True,
    ).eval()

    prompt = torch.randint(0, 64, (1, 4))
    next_token = torch.randint(0, 64, (1, 1))
    full_tokens = torch.cat([prompt, next_token], dim=1)

    with torch.inference_mode():
        full_logits = model(
            full_tokens,
            position_ids=torch.arange(5).unsqueeze(0),
        )

    cache = ContiguousKVCache(num_layers=len(model.layers))

    with torch.inference_mode():
        model(
            prompt,
            position_ids=torch.arange(4).unsqueeze(0),
            kv_cache=cache,
        )
        cached_logits = model(
            next_token,
            position_ids=torch.tensor([[4]]),
            kv_cache=cache,
        )

    assert all(layer.seq_len == 5 for layer in cache.layers)
    torch.testing.assert_close(
        cached_logits[:, -1],
        full_logits[:, -1],
        atol=1e-5,
        rtol=1e-5,
    )
