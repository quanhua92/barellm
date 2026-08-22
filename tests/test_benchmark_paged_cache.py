import torch

from barellm.engine.batched_kv_cache import BatchKVCache
from barellm.engine.block_pool import BlockPool
from barellm.engine.kv_cache_manager import KVCacheManager
from barellm.engine.paged_kv_cache import PagedKVCache
from barellm.engine.request import Request


def make_manager(
    lengths: list[int],
    block_size: int,
) -> tuple[KVCacheManager, list[Request], PagedKVCache, BlockPool]:
    max_blocks = sum((length + block_size - 1) // block_size for length in lengths)
    pool = BlockPool(max_blocks)
    storage = PagedKVCache(
        num_layers=2,
        max_blocks=max_blocks,
        num_kv_heads=2,
        block_size=block_size,
        head_dim=3,
    )
    manager = KVCacheManager(block_size, pool, storage)
    requests = []
    for index, length in enumerate(lengths):
        request = Request(
            id=f"benchmark-{index}",
            token_ids=torch.zeros((1, length), dtype=torch.long),
        )
        assert manager.allocate_request(request)
        requests.append(request)
    return manager, requests, storage, pool


def test_boundary_lengths_round_trip_through_physical_blocks() -> None:
    block_size = 16
    lengths = [15, 16, 17, 31, 32, 33]
    manager, requests, storage, pool = make_manager(lengths, block_size)

    try:
        for index, request in enumerate(requests):
            cache = manager.get_cache(request)
            key = torch.full((1, 2, request.seq_len, 3), float(index + 1))
            value = key + 100
            cache.layer(0).append(key, value)
            actual_key, actual_value = cache.layer(0).read()
            torch.testing.assert_close(actual_key, key)
            torch.testing.assert_close(actual_value, value)
    finally:
        for request in requests:
            manager.free_request(request.id)

    assert not storage.request_pages
    assert len(pool.free_ids) == len(pool.blocks)


def test_batched_read_pads_unequal_paged_histories() -> None:
    manager, requests, _storage, pool = make_manager([15, 33], block_size=16)
    caches = [manager.get_cache(request) for request in requests]

    try:
        expected = []
        for index, cache in enumerate(caches):
            key = torch.full((1, 2, requests[index].seq_len, 3), float(index + 1))
            value = key + 100
            cache.layer(0).append(key, value)
            expected.append((key, value))

        key, value = BatchKVCache(caches).layer(0).read()

        assert key.shape == (2, 2, 33, 3)
        assert value.shape == key.shape
        for row, (expected_key, expected_value) in enumerate(expected):
            length = expected_key.shape[2]
            torch.testing.assert_close(key[row : row + 1, :, :length, :], expected_key)
            torch.testing.assert_close(
                value[row : row + 1, :, :length, :], expected_value
            )
            if length < 33:
                assert torch.count_nonzero(key[row, :, length:, :]) == 0
                assert torch.count_nonzero(value[row, :, length:, :]) == 0
    finally:
        for request in requests:
            manager.free_request(request.id)

    assert len(pool.free_ids) == len(pool.blocks)
