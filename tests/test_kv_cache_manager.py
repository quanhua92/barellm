import pytest
import torch

from barellm.engine.block_pool import BlockPool
from barellm.engine.kv_cache_manager import KVCacheManager
from barellm.engine.paged_kv_cache import PagedKVCache
from barellm.engine.request import Request


def make_manager(max_blocks: int = 2) -> tuple[KVCacheManager, BlockPool]:
    pool = BlockPool(max_blocks)
    storage = PagedKVCache(
        num_layers=1,
        max_blocks=max_blocks,
        num_kv_heads=1,
        block_size=2,
        head_dim=2,
    )
    return KVCacheManager(2, pool, storage), pool


def make_request(request_id: str, seq_len: int) -> Request:
    return Request(
        id=request_id,
        token_ids=torch.zeros(1, seq_len, dtype=torch.long),
    )


def test_manager_allocates_grows_and_frees_request_cache() -> None:
    manager, pool = make_manager()
    request = make_request("A", seq_len=2)

    assert manager.allocate_request(request)
    assert len(manager.request_id_to_blocks[request.id]) == 1
    manager.get_cache(request)

    request.append(torch.zeros(1, 1, dtype=torch.long))
    assert manager.allocate_request(request)
    assert len(manager.request_id_to_blocks[request.id]) == 2
    assert not pool.free_ids

    manager.free_request(request.id)

    assert len(pool.free_ids) == 2
    with pytest.raises(KeyError, match="Unknown request"):
        manager.get_cache(request)


def test_manager_reuses_freed_blocks_for_new_request() -> None:
    manager, pool = make_manager()
    request_a = make_request("A", seq_len=2)
    request_b = make_request("B", seq_len=2)

    assert manager.allocate_request(request_a)
    manager.free_request(request_a.id)

    assert manager.allocate_request(request_b)
    assert len(manager.request_id_to_blocks[request_b.id]) == 1
    assert len(pool.free_ids) == 1


def test_reused_block_does_not_expose_previous_logical_cache() -> None:
    manager, _pool = make_manager(max_blocks=1)
    request_a = make_request("A", seq_len=1)
    request_b = make_request("B", seq_len=1)

    assert manager.allocate_request(request_a)
    block_a = manager.request_id_to_blocks[request_a.id][0].block_id
    old_key = torch.full((1, 1, 1, 2), 11.0)
    old_value = torch.full((1, 1, 1, 2), 22.0)
    manager.get_cache(request_a).layer(0).append(old_key, old_value)
    manager.free_request(request_a.id)

    assert manager.allocate_request(request_b)
    block_b = manager.request_id_to_blocks[request_b.id][0].block_id
    assert block_b == block_a

    new_layer = manager.get_cache(request_b).layer(0)
    with pytest.raises(ValueError, match="empty"):
        new_layer.read()

    new_key = torch.full((1, 1, 1, 2), 33.0)
    new_value = torch.full((1, 1, 1, 2), 44.0)
    new_layer.append(new_key, new_value)
    torch.testing.assert_close(new_layer.read(), (new_key, new_value))
