import torch

from barellm.engine.batched_kv_cache import BatchKVCache
from barellm.engine.block_pool import BlockPool
from barellm.engine.contiguous_kv_cache import ContiguousKVCache
from barellm.engine.kv_cache_manager import KVCacheManager
from barellm.engine.paged_kv_cache import PagedKVCache
from barellm.engine.request import Request
from barellm.models.qwen3 import Qwen3ForCausalLM


def test_batch_cache_pads_unequal_histories_and_masks_padding() -> None:
    cache_a = ContiguousKVCache(num_layers=1)
    cache_b = ContiguousKVCache(num_layers=1)

    cache_a.layer(0).append(
        torch.randn(1, 2, 2, 3),
        torch.randn(1, 2, 2, 3),
    )
    cache_b.layer(0).append(
        torch.randn(1, 2, 4, 3),
        torch.randn(1, 2, 4, 3),
    )

    batch_layer = BatchKVCache([cache_a, cache_b]).layer(0)

    key, value = batch_layer.append(
        torch.randn(2, 2, 1, 3),
        torch.randn(2, 2, 1, 3),
    )

    assert key.shape == (2, 2, 5, 3)
    assert value.shape == (2, 2, 5, 3)
    assert batch_layer.seq_len == 5

    mask = batch_layer.attention_mask(q_len=1)
    assert mask is not None

    assert mask.shape == (2, 1, 1, 5)
    assert mask[0, 0, 0].tolist() == [True, True, True, False, False]
    assert mask[1, 0, 0].tolist() == [True, True, True, True, True]


def make_qwen3() -> Qwen3ForCausalLM:
    return Qwen3ForCausalLM(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        head_dim=4,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        use_qk_norm=True,
    ).eval()


def prepare_requests(
    model: Qwen3ForCausalLM,
    prompts: list[torch.Tensor],
    next_tokens: list[torch.Tensor],
) -> tuple[KVCacheManager, list[Request]]:
    block_size = 2
    pool = BlockPool(16)
    storage = PagedKVCache(
        num_layers=2,
        max_blocks=16,
        num_kv_heads=2,
        block_size=block_size,
        head_dim=4,
    )
    manager = KVCacheManager(block_size, pool, storage)
    requests = [
        Request(id=f"request-{index}", token_ids=prompt.clone())
        for index, prompt in enumerate(prompts)
    ]

    for request, next_token in zip(requests, next_tokens):
        assert manager.allocate_request(request)
        position_ids = torch.arange(request.seq_len).unsqueeze(0)
        model(
            request.token_ids,
            position_ids=position_ids,
            kv_cache=manager.get_cache(request),
        )
        request.append(next_token.clone())

    return manager, requests


def test_real_batched_paged_decode_matches_independent_decodes() -> None:
    torch.manual_seed(0)
    model = make_qwen3()
    prompts = [
        torch.tensor([[1, 2]]),
        torch.tensor([[3, 4, 5, 6]]),
    ]
    next_tokens = [torch.tensor([[7]]), torch.tensor([[8]])]

    with torch.inference_mode():
        independent_manager, independent_requests = prepare_requests(
            model, prompts, next_tokens
        )
        independent_logits = []
        for request in independent_requests:
            assert independent_manager.allocate_request(request)
            logits = model(
                request.token_ids[:, -1:],
                position_ids=torch.tensor([[request.seq_len - 1]]),
                kv_cache=independent_manager.get_cache(request),
            )
            independent_logits.append(logits)

        batched_manager, batched_requests = prepare_requests(
            model, prompts, next_tokens
        )
        for request in batched_requests:
            assert batched_manager.allocate_request(request)

        input_ids = torch.cat(
            [request.token_ids[:, -1:] for request in batched_requests],
            dim=0,
        )
        position_ids = torch.tensor(
            [[request.seq_len - 1] for request in batched_requests]
        )
        batch_cache = BatchKVCache(
            [batched_manager.get_cache(request) for request in batched_requests]
        )
        batched_logits = model(
            input_ids,
            position_ids=position_ids,
            kv_cache=batch_cache,
        )

    for row, expected in enumerate(independent_logits):
        torch.testing.assert_close(
            batched_logits[row : row + 1],
            expected,
            atol=1e-5,
            rtol=1e-5,
        )
