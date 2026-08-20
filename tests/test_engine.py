import torch
from torch import nn

from barellm.engine.block_pool import BlockPool
from barellm.engine.engine import Engine
from barellm.engine.kv_cache_manager import KVCacheManager
from barellm.engine.paged_kv_cache import PagedKVCache
from barellm.engine.request import Request
from barellm.engine.scheduler import Scheduler


class TinyEngineModel(nn.Module):
    """Small model implementing the Engine's forward contract."""

    def __init__(self, vocab_size: int = 64, hidden_size: int = 16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        token_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        kv_cache=None,
        request_ids: list[str] | None = None,
    ) -> torch.Tensor:
        del position_ids, kv_cache, request_ids
        return self.lm_head(self.embedding(token_ids))


class TestEngine:
    def test_simple(self):
        scheduler = Scheduler(max_batch=2)
        request = Request(
            id="A",
            token_ids=torch.tensor([[20, 30, 40]]),
            max_new_tokens=2,
        )
        scheduler.add_request(request)
        block_pool = BlockPool(128)
        block_size = 16
        paged_kv_cache = PagedKVCache()
        kv_cache_manager = KVCacheManager(block_size, block_pool, paged_kv_cache)
        model = TinyEngineModel()
        engine = Engine(model, scheduler, kv_cache_manager)
        engine.run(max_steps=10)

        assert request.generated_count == 2
        assert request.finish_reason == "length"
        assert request.id not in kv_cache_manager.request_id_to_blocks
