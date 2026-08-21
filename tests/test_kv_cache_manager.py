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
