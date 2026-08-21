import pytest
import torch
from torch import nn

from barellm.engine.block_pool import BlockPool
from barellm.engine.engine import Engine
from barellm.engine.kv_cache_manager import KVCacheManager
from barellm.engine.paged_kv_cache import PagedKVCache
from barellm.engine.request import Request
from barellm.engine.scheduler import Scheduler
from barellm.models.qwen3 import Qwen3ForCausalLM


class TinyEngineModel(nn.Module):
    """Small model implementing the Engine's forward contract."""

    def __init__(self, vocab_size: int = 64, hidden_size: int = 16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        self.forward_calls = 0

    def forward(
        self,
        token_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        kv_cache=None,
        request_ids: list[str] | None = None,
    ) -> torch.Tensor:
        self.forward_calls += 1
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
        paged_kv_cache = PagedKVCache(
            num_layers=1,
            max_blocks=128,
            num_kv_heads=1,
            block_size=block_size,
            head_dim=1,
        )
        kv_cache_manager = KVCacheManager(
            block_size,
            block_pool,
            paged_kv_cache,
        )
        model = TinyEngineModel()
        engine = Engine(model, scheduler, kv_cache_manager)
        engine.run(max_steps=10)

        assert request.generated_count == 2
        assert request.finish_reason == "length"
        assert request.id not in kv_cache_manager.request_id_to_blocks

    def test_batches_requests_with_different_prompt_lengths(self):
        model = Qwen3ForCausalLM(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            head_dim=4,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            use_qk_norm=True,
        )
        scheduler = Scheduler(max_batch=2)
        requests = [
            Request(
                id="short",
                token_ids=torch.tensor([[1, 2]]),
                max_new_tokens=2,
                temperature=0.0,
            ),
            Request(
                id="long",
                token_ids=torch.tensor([[1, 2, 3, 4]]),
                max_new_tokens=2,
                temperature=0.0,
            ),
        ]
        for request in requests:
            scheduler.add_request(request)

        block_size = 2
        pool = BlockPool(16)
        paged_kv_cache = PagedKVCache(
            num_layers=2,
            max_blocks=16,
            num_kv_heads=2,
            block_size=block_size,
            head_dim=4,
        )
        kv_cache_manager = KVCacheManager(
            block_size,
            pool,
            paged_kv_cache,
        )
        engine = Engine(model, scheduler, kv_cache_manager)

        engine.run(max_steps=2)

        for request in requests:
            assert request.generated_count == 2
            assert request.finish_reason == "length"
            assert request.id not in kv_cache_manager.request_id_to_blocks

    def test_decode_skips_request_when_no_block_is_available(self):
        model = TinyEngineModel()
        scheduler = Scheduler(max_batch=1)
        request = Request(
            id="blocked",
            token_ids=torch.tensor([[20, 30]]),
            max_new_tokens=2,
        )
        scheduler.add_request(request)
        block_size = 2
        block_pool = BlockPool(1)
        paged_kv_cache = PagedKVCache(
            num_layers=1,
            max_blocks=1,
            num_kv_heads=1,
            block_size=block_size,
            head_dim=1,
        )
        kv_cache_manager = KVCacheManager(
            block_size,
            block_pool,
            paged_kv_cache,
        )
        assert kv_cache_manager.allocate_request(request)
        request.append(torch.tensor([[40]]))
        scheduler.pop_request(request)
        scheduler.start_request(request)
        engine = Engine(model, scheduler, kv_cache_manager)

        engine._decode()

        assert request.generated_count == 1
        assert model.forward_calls == 0
        assert request.status.value == "running"

    @pytest.mark.parametrize(
        ("max_new_tokens", "expected_generated_count"),
        [(0, 0), (1, 1)],
    )
    def test_request_finishing_during_prefill_releases_cache(
        self,
        max_new_tokens: int,
        expected_generated_count: int,
    ) -> None:
        model = TinyEngineModel()
        scheduler = Scheduler(max_batch=1)
        request = Request(
            id=f"prefill-finish-{max_new_tokens}",
            token_ids=torch.tensor([[20, 30]]),
            max_new_tokens=max_new_tokens,
            temperature=0.0,
        )
        scheduler.add_request(request)

        block_size = 2
        block_pool = BlockPool(2)
        paged_kv_cache = PagedKVCache(
            num_layers=1,
            max_blocks=2,
            num_kv_heads=1,
            block_size=block_size,
            head_dim=1,
        )
        kv_cache_manager = KVCacheManager(
            block_size,
            block_pool,
            paged_kv_cache,
        )
        engine = Engine(model, scheduler, kv_cache_manager)

        engine.run(max_steps=2)

        assert request.status.value == "finished"
        assert request.generated_count == expected_generated_count
        assert request.id not in kv_cache_manager.request_id_to_blocks
        assert request.id not in paged_kv_cache.request_pages
        assert len(block_pool.free_ids) == len(block_pool.blocks)
