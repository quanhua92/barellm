import time
from typing import Any, cast

import pytest
import torch
from torch import nn

from barellm.engine.block_pool import BlockPool
from barellm.engine.engine import Engine
from barellm.engine.generate import GenerationResult, generate
from barellm.engine.kv_cache_manager import KVCacheManager
from barellm.engine.paged_kv_cache import PagedKVCache
from barellm.engine.request import Request
from barellm.engine.scheduler import Scheduler
from barellm.models.qwen3 import Qwen3ForCausalLM


class FixedTokenModel(nn.Module):
    def __init__(self, token: int, vocab_size: int = 16) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.token = token
        self.vocab_size = vocab_size
        self.input_lengths: list[int] = []
        self.position_ids: list[list[int] | None] = []
        self.cache_was_none: list[bool] = []

    def forward(
        self,
        token_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        kv_cache=None,
        request_ids: list[str] | None = None,
    ) -> torch.Tensor:
        del request_ids
        self.input_lengths.append(token_ids.shape[1])
        self.position_ids.append(
            None if position_ids is None else position_ids[0].tolist()
        )
        self.cache_was_none.append(kv_cache is None)
        logits = torch.full(
            (*token_ids.shape, self.vocab_size),
            float("-inf"),
            device=token_ids.device,
        )
        logits[..., self.token] = 0.0
        return logits


def make_engine() -> Engine:
    scheduler = Scheduler(max_batch=1)
    block_size = 2
    block_pool = BlockPool(4)
    paged_kv_cache = PagedKVCache(
        num_layers=1,
        max_blocks=4,
        num_kv_heads=1,
        block_size=block_size,
        head_dim=1,
    )
    manager = KVCacheManager(
        block_size,
        block_pool,
        paged_kv_cache,
    )
    return Engine(FixedTokenModel(5), scheduler, manager)


def make_uncached_engine(model: nn.Module | None = None) -> Engine:
    return Engine(
        model or FixedTokenModel(5),
        Scheduler(max_batch=1),
        None,
    )


def make_qwen_engine(
    model: Qwen3ForCausalLM,
    cached: bool,
    num_kv_heads: int,
) -> Engine:
    scheduler = Scheduler(max_batch=1)
    if not cached:
        return Engine(model, scheduler, None)

    block_size = 2
    block_pool = BlockPool(8)
    paged_kv_cache = PagedKVCache(
        num_layers=len(model.layers),
        max_blocks=8,
        num_kv_heads=num_kv_heads,
        block_size=block_size,
        head_dim=4,
    )
    manager = KVCacheManager(block_size, block_pool, paged_kv_cache)
    return Engine(model, scheduler, manager)


def test_generate_returns_result_and_generated_view() -> None:
    result = generate(
        make_engine(),
        torch.tensor([[1, 2]]),
        max_new_tokens=1,
        temperature=0.0,
    )

    assert isinstance(result, GenerationResult)
    assert result.token_ids.tolist() == [[1, 2, 5]]
    assert result.generated_token_ids.tolist() == [[5]]
    assert result.prompt_length == 2
    assert result.generated_count == 1
    assert result.finish_reason == "length"


def test_generate_forwards_deadline() -> None:
    result = generate(
        make_engine(),
        torch.tensor([[1, 2]]),
        max_new_tokens=4,
        temperature=0.0,
        deadline=time.monotonic() - 1.0,
    )

    assert result.finish_reason == "abort"
    assert result.stop_reason is None


def test_generate_uncached_recomputes_the_full_sequence() -> None:
    model = FixedTokenModel(5)
    result = generate(
        make_uncached_engine(model),
        torch.tensor([[1, 2]]),
        max_new_tokens=2,
        temperature=0.0,
        use_cache=False,
    )

    assert result.token_ids.tolist() == [[1, 2, 5, 5]]
    assert model.input_lengths == [2, 3]
    assert model.position_ids == [[0, 1], [0, 1, 2]]
    assert model.cache_was_none == [True, True]


def test_generate_requires_cache_manager_for_cached_mode() -> None:
    with pytest.raises(ValueError, match="KV cache manager"):
        generate(
            make_uncached_engine(),
            torch.tensor([[1, 2]]),
            max_new_tokens=1,
            use_cache=True,
        )


@pytest.mark.parametrize(
    "num_kv_heads",
    [4, 2, 1],
    ids=["mha", "gqa", "mqa"],
)
def test_cached_and_uncached_generation_match(
    num_kv_heads: int,
) -> None:
    torch.manual_seed(0)
    cached_model = Qwen3ForCausalLM(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        head_dim=4,
        num_layers=2,
        num_heads=4,
        num_kv_heads=num_kv_heads,
        use_qk_norm=True,
    ).eval()
    uncached_model = Qwen3ForCausalLM(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        head_dim=4,
        num_layers=2,
        num_heads=4,
        num_kv_heads=num_kv_heads,
        use_qk_norm=True,
    ).eval()
    uncached_model.load_state_dict(cached_model.state_dict())

    prompt = torch.tensor([[1, 2, 3]])
    with torch.inference_mode():
        cached = generate(
            make_qwen_engine(cached_model, cached=True, num_kv_heads=num_kv_heads),
            prompt,
            max_new_tokens=3,
            temperature=0.0,
            eos_ids=set(),
            use_cache=True,
        )
        uncached = generate(
            make_qwen_engine(
                uncached_model,
                cached=False,
                num_kv_heads=num_kv_heads,
            ),
            prompt,
            max_new_tokens=3,
            temperature=0.0,
            eos_ids=set(),
            use_cache=False,
        )

    assert cached.finish_reason == uncached.finish_reason == "length"
    assert cached.generated_count == uncached.generated_count == 3
    torch.testing.assert_close(cached.token_ids, uncached.token_ids)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_new_tokens": -1}, "max_new_tokens"),
        ({"temperature": -1.0}, "temperature"),
        ({"top_k": -1}, "top_k"),
        ({"top_p": 0.0}, "top_p"),
        ({"top_p": 1.1}, "top_p"),
    ],
)
def test_generate_rejects_invalid_sampling_arguments(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        generate(
            make_engine(),
            torch.tensor([[1, 2]]),
            **cast(Any, kwargs),
        )


def test_generate_rejects_invalid_token_ids() -> None:
    with pytest.raises(ValueError, match="shape"):
        generate(make_engine(), torch.tensor([1, 2]))

    with pytest.raises(ValueError, match="batch size"):
        generate(make_engine(), torch.tensor([[1], [2]]))

    with pytest.raises(ValueError, match="at least one"):
        generate(make_engine(), torch.empty((1, 0), dtype=torch.long))

    with pytest.raises(ValueError, match="integer"):
        generate(make_engine(), torch.tensor([[1.0, 2.0]]))


def test_generate_requires_decoder_for_stop_strings() -> None:
    with pytest.raises(ValueError, match="decode_fn"):
        generate(
            make_engine(),
            torch.tensor([[1, 2]]),
            stop_strings=["<stop>"],
        )


def test_generate_requires_idle_engine() -> None:
    engine = make_engine()
    engine.scheduler.add_request(
        Request(
            id="already-running",
            token_ids=torch.tensor([[1, 2]]),
        )
    )

    with pytest.raises(ValueError, match="idle engine"):
        generate(engine, torch.tensor([[1, 2]]))
