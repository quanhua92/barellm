from typing import cast

import pytest
import torch

from barellm.engine.batched_kv_cache import BatchKVCache, BatchLayerKV
from barellm.engine.block_pool import BlockPool
from barellm.engine.paged_kv_cache import PagedKVCache, PagedLayerKV


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
    layer = cast(PagedLayerKV, storage.get_cache("A").layer(0))

    key = torch.arange(18, dtype=torch.float32).reshape(1, 2, 3, 3)
    value = key + 100

    returned_key, returned_value = layer.append(key, value)

    assert layer.seq_len == 3
    torch.testing.assert_close(returned_key, key)
    torch.testing.assert_close(returned_value, value)
    torch.testing.assert_close(layer.read(), (key, value))


def test_append_only_matches_legacy_append_and_exposes_metadata() -> None:
    storage, pool = make_storage(block_size=2)
    storage.register_request("A", pool.allocate(2))
    layer = cast(PagedLayerKV, storage.get_cache("A").layer(0))

    key = torch.arange(18, dtype=torch.float32).reshape(1, 2, 3, 3)
    value = key + 100
    layer.append_only(key, value)

    metadata = layer.paged_metadata_post_append()
    assert metadata.seq_len == 3
    assert metadata.block_size == 2
    assert metadata.layer_idx == 0
    assert metadata.block_table.dtype == torch.int32
    torch.testing.assert_close(layer.read(), (key, value))


def test_batched_paged_metadata_is_rectangular_and_device_resident() -> None:
    storage, pool = make_storage(block_size=2, max_blocks=4)
    storage.register_request("A", pool.allocate(1))
    storage.register_request("B", pool.allocate(2))
    batch = BatchKVCache([storage.get_cache("A"), storage.get_cache("B")])
    layer = cast(BatchLayerKV, batch.layer(0))
    layer.append_only(
        torch.randn(2, 2, 1, 3),
        torch.randn(2, 2, 1, 3),
    )

    metadata = layer.paged_metadata_post_append()
    assert metadata.block_tables.shape == (2, 2)
    assert metadata.block_tables.dtype == torch.int32
    assert metadata.seq_lens.tolist() == [1, 1]
    assert metadata.block_tables.device == storage.device


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


def test_paged_block_table_refreshes_when_request_grows() -> None:
    storage, pool = make_storage(block_size=2, max_blocks=2)
    initial_blocks = pool.allocate(1)
    storage.register_request("A", initial_blocks)
    layer = storage.get_cache("A").layer(0)

    key = torch.arange(18, dtype=torch.float32).reshape(1, 2, 3, 3)
    value = key + 100
    layer.append(key[:, :, :2], value[:, :, :2])

    expanded_blocks = initial_blocks + pool.allocate(1)
    storage.register_request("A", expanded_blocks)
    layer.append(key[:, :, 2:], value[:, :, 2:])

    assert storage.request_pages["A"].block_table.tolist() == [
        block.block_id for block in expanded_blocks
    ]
    actual_key, actual_value = layer.read()
    assert actual_key.shape == (1, 2, 3, 3)
    assert actual_value.shape == (1, 2, 3, 3)
    torch.testing.assert_close(actual_key, key)
    torch.testing.assert_close(actual_value, value)


def test_paged_decode_appends_across_multiple_boundaries() -> None:
    storage, pool = make_storage(block_size=2, max_blocks=3)
    storage.register_request("A", pool.allocate(3))
    layer = storage.get_cache("A").layer(0)

    key = torch.arange(24, dtype=torch.float32).reshape(1, 2, 4, 3)
    value = key + 100

    layer.append(key[:, :, :2], value[:, :, :2])
    layer.append(key[:, :, 2:3], value[:, :, 2:3])
    layer.append(key[:, :, 3:4], value[:, :, 3:4])

    assert layer.seq_len == 4
    torch.testing.assert_close(layer.read(), (key, value))


def test_paged_requests_do_not_share_values() -> None:
    storage, pool = make_storage(max_blocks=2)
    storage.register_request("A", pool.allocate(1))
    storage.register_request("B", pool.allocate(1))

    key_a = torch.ones(1, 2, 1, 3)
    value_a = torch.ones(1, 2, 1, 3)
    key_b = torch.full((1, 2, 1, 3), 2.0)
    value_b = torch.full((1, 2, 1, 3), 2.0)

    layer_a = storage.get_cache("A").layer(0)
    layer_b = storage.get_cache("B").layer(0)
    layer_a.append(key_a, value_a)
    layer_b.append(key_b, value_b)

    torch.testing.assert_close(layer_a.read(), (key_a, value_a))
    torch.testing.assert_close(layer_b.read(), (key_b, value_b))


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
