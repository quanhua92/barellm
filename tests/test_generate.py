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


class FixedTokenModel(nn.Module):
    def __init__(self, token: int, vocab_size: int = 16) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.token = token
        self.vocab_size = vocab_size

    def forward(
        self,
        token_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        kv_cache=None,
        request_ids: list[str] | None = None,
    ) -> torch.Tensor:
        del position_ids, kv_cache, request_ids
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
