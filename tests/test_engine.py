import time

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
from barellm.sampling.stops import FINISH_ABORT, FINISH_STOP


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


class FixedTokenModel(nn.Module):
    """Emit a predetermined token on each forward call."""

    def __init__(self, tokens: list[int], vocab_size: int = 16):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.tokens = tokens
        self.vocab_size = vocab_size
        self.forward_calls = 0

    def forward(
        self,
        token_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        kv_cache=None,
        request_ids: list[str] | None = None,
    ) -> torch.Tensor:
        del position_ids, kv_cache, request_ids
        token = self.tokens[min(self.forward_calls, len(self.tokens) - 1)]
        self.forward_calls += 1
        logits = torch.full(
            (*token_ids.shape, self.vocab_size),
            float("-inf"),
            device=token_ids.device,
        )
        logits[..., token] = 0.0
        return logits


def make_test_engine(
    model: nn.Module,
    request: Request,
) -> tuple[Engine, KVCacheManager, BlockPool, PagedKVCache]:
    scheduler = Scheduler(max_batch=1)
    scheduler.add_request(request)
    block_size = 2
    block_pool = BlockPool(4)
    paged_kv_cache = PagedKVCache(
        num_layers=1,
        max_blocks=4,
        num_kv_heads=1,
        block_size=block_size,
        head_dim=1,
    )
    kv_cache_manager = KVCacheManager(
        block_size,
        block_pool,
        paged_kv_cache,
    )
    return (
        Engine(model, scheduler, kv_cache_manager),
        kv_cache_manager,
        block_pool,
        paged_kv_cache,
    )


def assert_cache_released(
    request: Request,
    manager: KVCacheManager,
    pool: BlockPool,
    paged_kv_cache: PagedKVCache,
) -> None:
    assert request.id not in manager.request_id_to_blocks
    assert request.id not in paged_kv_cache.request_pages
    assert len(pool.free_ids) == len(pool.blocks)


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

    def test_eos_during_prefill_calls_callbacks_once_and_releases_cache(self):
        streamed = []
        finished = []
        request = Request(
            id="eos-prefill",
            token_ids=torch.tensor([[1, 2]]),
            max_new_tokens=4,
            eos_ids={7},
            on_token=lambda token_id, count: streamed.append((token_id, count)),
            on_finish=lambda reason, stop: finished.append((reason, stop)),
        )
        model = FixedTokenModel([7])
        engine, manager, pool, paged_kv_cache = make_test_engine(model, request)

        engine.run(max_steps=2)

        assert request.finish_reason == FINISH_STOP
        assert request.stop_reason == 7
        assert streamed == [(7, 3)]
        assert finished == [(FINISH_STOP, 7)]
        assert model.forward_calls == 1
        assert_cache_released(request, manager, pool, paged_kv_cache)

    def test_eos_during_decode_calls_callbacks_once_and_releases_cache(self):
        streamed = []
        finished = []
        request = Request(
            id="eos-decode",
            token_ids=torch.tensor([[1, 2]]),
            max_new_tokens=4,
            eos_ids={7},
            on_token=lambda token_id, count: streamed.append((token_id, count)),
            on_finish=lambda reason, stop: finished.append((reason, stop)),
        )
        model = FixedTokenModel([5, 7])
        engine, manager, pool, paged_kv_cache = make_test_engine(model, request)

        engine.run(max_steps=3)

        assert request.finish_reason == FINISH_STOP
        assert request.stop_reason == 7
        assert streamed == [(5, 3), (7, 4)]
        assert finished == [(FINISH_STOP, 7)]
        assert model.forward_calls == 2
        assert_cache_released(request, manager, pool, paged_kv_cache)

    def test_callback_abort_calls_finish_once_and_releases_cache(self):
        finished = []
        request = Request(
            id="callback-abort",
            token_ids=torch.tensor([[1, 2]]),
            max_new_tokens=4,
            on_token=lambda token_id, count: False,
            on_finish=lambda reason, stop: finished.append((reason, stop)),
        )
        model = FixedTokenModel([5])
        engine, manager, pool, paged_kv_cache = make_test_engine(model, request)

        engine.run(max_steps=2)

        assert request.finish_reason == FINISH_ABORT
        assert request.stop_reason is None
        assert finished == [(FINISH_ABORT, None)]
        assert_cache_released(request, manager, pool, paged_kv_cache)

    def test_stop_string_calls_finish_once_and_releases_cache(self):
        finished = []
        request = Request(
            id="stop-string",
            token_ids=torch.tensor([[1, 2]]),
            max_new_tokens=4,
            stop_strings=["<stop>"],
            decode_fn=lambda token_ids: "hello<stop>",
            on_finish=lambda reason, stop: finished.append((reason, stop)),
        )
        model = FixedTokenModel([5])
        engine, manager, pool, paged_kv_cache = make_test_engine(model, request)

        engine.run(max_steps=2)

        assert request.finish_reason == FINISH_STOP
        assert request.stop_reason == "<stop>"
        assert finished == [(FINISH_STOP, "<stop>")]
        assert_cache_released(request, manager, pool, paged_kv_cache)

    def test_expired_deadline_aborts_and_releases_cache(self):
        finished = []
        request = Request(
            id="deadline",
            token_ids=torch.tensor([[1, 2]]),
            max_new_tokens=4,
            deadline=time.monotonic() - 1.0,
            on_finish=lambda reason, stop: finished.append((reason, stop)),
        )
        model = FixedTokenModel([5])
        engine, manager, pool, paged_kv_cache = make_test_engine(model, request)

        engine.run(max_steps=2)

        assert request.finish_reason == FINISH_ABORT
        assert request.stop_reason is None
        assert finished == [(FINISH_ABORT, None)]
        assert_cache_released(request, manager, pool, paged_kv_cache)
