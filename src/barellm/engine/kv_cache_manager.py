from collections import defaultdict

from barellm.engine.block_pool import BlockPool, KVCacheBlock
from barellm.engine.paged_kv_cache import PagedKVCache
from barellm.engine.request import Request
from barellm.models.cache import KVCache


class KVCacheManager:
    def __init__(
        self,
        block_size: int,
        block_pool: BlockPool,
        paged_kv_cache: PagedKVCache,
    ):
        self.block_size = block_size
        self.block_pool = block_pool
        self.paged_kv_cache = paged_kv_cache
        self.request_id_to_blocks: dict[str, list[KVCacheBlock]] = defaultdict(list)

    def get_cache(self, req: Request) -> KVCache:
        request_id = req.id
        if request_id not in self.request_id_to_blocks:
            raise KeyError(f"Unknown request: {request_id}")
        return self.paged_kv_cache.get_cache(request_id)

    def allocate_request(self, req: Request) -> bool:
        block_ids = self.request_id_to_blocks.get(req.id, [])
        num_blocks = int((req.seq_len + self.block_size - 1) / self.block_size)
        remaining_blocks = max(num_blocks - len(block_ids), 0)
        if self.block_pool.can_allocate(remaining_blocks) is False:
            return False
        new_ids = self.block_pool.allocate(remaining_blocks)
        self.request_id_to_blocks[req.id].extend(new_ids)
        self.paged_kv_cache.register_request(
            req.id,
            self.request_id_to_blocks[req.id],
        )
        return True

    def free_request(self, request_id: str) -> None:
        blocks = self.request_id_to_blocks.pop(request_id, None)
        if blocks is not None:
            self.block_pool.free(blocks)
        self.paged_kv_cache.unregister_request(request_id)
