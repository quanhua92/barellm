import pytest
import torch

from barellm.engine.block_pool import BlockPool
from barellm.engine.paged_kv_cache import PagedKVCache


def make_storage(
    num_layers: int = 2,
    max_blocks: int = 4,
    num_kv_heads: int = 2,
    block_size: int = 2,
    head_dim: int = 3,
) -> tuple[PagedKVCache, BlockPool]:
    storage = PagedKVCache(
        num_layers=num_layers,
        max_blocks=max_blocks,
        num_kv_heads=num_kv_heads,
        block_size=block_size,
        head_dim=head_dim,
    )
    return storage, BlockPool(max_blocks)


def test_paged_append_gathers_across_block_boundary() -> None:
    storage, pool = make_storage(block_size=2)
    storage.register_request("A", pool.allocate(2))
    layer = storage.get_cache("A").layer(0)

    key = torch.arange(18, dtype=torch.float32).reshape(1, 2, 3, 3)
    value = key + 100

    returned_key, returned_value = layer.append(key, value)

    assert layer.seq_len == 3
    torch.testing.assert_close(returned_key, key)
    torch.testing.assert_close(returned_value, value)
    torch.testing.assert_close(layer.read(), (key, value))


def test_paged_append_writes_new_token_after_boundary() -> None:
    storage, pool = make_storage(block_size=2)
    storage.register_request("A", pool.allocate(2))
    layer = storage.get_cache("A").layer(0)

    key_a = torch.randn(1, 2, 2, 3)
    value_a = torch.randn(1, 2, 2, 3)
    key_b = torch.randn(1, 2, 1, 3)
    value_b = torch.randn(1, 2, 1, 3)

    layer.append(key_a, value_a)
    layer.append(key_b, value_b)

    assert layer.seq_len == 3
    torch.testing.assert_close(
        layer.read(),
        (torch.cat([key_a, key_b], dim=2), torch.cat([value_a, value_b], dim=2)),
    )


def test_paged_layers_have_independent_storage_and_lengths() -> None:
    storage, pool = make_storage()
    storage.register_request("A", pool.allocate(2))
    cache = storage.get_cache("A")
    key = torch.randn(1, 2, 1, 3)
    value = torch.randn(1, 2, 1, 3)

    cache.layer(0).append(key, value)

    assert cache.layer(0).seq_len == 1
    assert cache.layer(1).seq_len == 0
    with pytest.raises(ValueError, match="empty"):
        cache.layer(1).read()


def test_paged_cache_rejects_insufficient_blocks() -> None:
    storage, pool = make_storage(block_size=2)
    storage.register_request("A", pool.allocate(1))
    layer = storage.get_cache("A").layer(0)

    key = torch.randn(1, 2, 3, 3)
    value = torch.randn(1, 2, 3, 3)

    with pytest.raises(ValueError, match="enough allocated"):
        layer.append(key, value)


def test_unregister_removes_request_cache() -> None:
    storage, pool = make_storage()
    storage.register_request("A", pool.allocate(1))

    storage.unregister_request("A")

    with pytest.raises(KeyError, match="Unknown request"):
        storage.get_cache("A")
