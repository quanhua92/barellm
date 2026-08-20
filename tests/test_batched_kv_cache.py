import torch

from barellm.engine.batched_kv_cache import BatchKVCache
from barellm.engine.contiguous_kv_cache import ContiguousKVCache


def test_batch_cache_pads_unequal_histories_and_masks_padding() -> None:
    cache_a = ContiguousKVCache(num_layers=1)
    cache_b = ContiguousKVCache(num_layers=1)

    cache_a.layer(0).append(
        torch.randn(1, 2, 2, 3),
        torch.randn(1, 2, 2, 3),
    )
    cache_b.layer(0).append(
        torch.randn(1, 2, 4, 3),
        torch.randn(1, 2, 4, 3),
    )

    batch_layer = BatchKVCache([cache_a, cache_b]).layer(0)

    key, value = batch_layer.append(
        torch.randn(2, 2, 1, 3),
        torch.randn(2, 2, 1, 3),
    )

    assert key.shape == (2, 2, 5, 3)
    assert value.shape == (2, 2, 5, 3)
    assert batch_layer.seq_len == 5

    mask = batch_layer.attention_mask(q_len=1)
    assert mask is not None

    assert mask.shape == (2, 1, 1, 5)
    assert mask[0, 0, 0].tolist() == [True, True, True, False, False]
    assert mask[1, 0, 0].tolist() == [True, True, True, True, True]
