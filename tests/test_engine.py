import torch

from barellm.engine.block_pool import BlockPool
from barellm.engine.engine import Engine
from barellm.engine.kv_cache_manager import KVCacheManager
from barellm.engine.paged_kv_cache import PagedKVCache
from barellm.engine.request import Request
from barellm.engine.scheduler import Scheduler


class TestEngine:
    def test_simple(self):
        scheduler = Scheduler(max_batch=2)
        scheduler.add_request(Request(id="A", token_ids=torch.tensor([[20, 30, 40]])))
        block_pool = BlockPool(128)
        block_size = 16
        paged_kv_cache = PagedKVCache()
        kv_cache_manager = KVCacheManager(block_size, block_pool, paged_kv_cache)
        engine = Engine(scheduler, kv_cache_manager)
        engine.run(max_steps=10)
        assert False
