import pytest
import torch

from barellm.engine.contiguous_kv_cache import ContiguousKVCache, ContiguousLayerKV


def make_kv(
    batch: int = 1,
    heads: int = 2,
    tokens: int = 3,
    head_dim: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    shape = (batch, heads, tokens, head_dim)
    return torch.randn(shape), torch.randn(shape)


def test_first_append_initializes_cache() -> None:
    cache = ContiguousLayerKV()
    key, value = make_kv(tokens=3)

    returned_key, returned_value = cache.append(key, value)

    assert cache.seq_len == 3
    assert returned_key is key
    assert returned_value is value
    torch.testing.assert_close(cache.read(), (key, value))


def test_append_returns_complete_history() -> None:
    cache = ContiguousLayerKV()
    key_a, value_a = make_kv(tokens=3)
    key_b, value_b = make_kv(tokens=1)

    cache.append(key_a, value_a)
    returned_key, returned_value = cache.append(key_b, value_b)

    expected_key = torch.cat([key_a, key_b], dim=2)
    expected_value = torch.cat([value_a, value_b], dim=2)

    assert cache.seq_len == 4
    torch.testing.assert_close(returned_key, expected_key)
    torch.testing.assert_close(returned_value, expected_value)
    torch.testing.assert_close(cache.read(), (expected_key, expected_value))


def test_layers_are_independent() -> None:
    cache = ContiguousKVCache(num_layers=2)
    key, value = make_kv(tokens=2)

    cache.layer(0).append(key, value)

    assert cache.layer(0).seq_len == 2
    assert cache.layer(1).seq_len == 0


def test_read_empty_cache_raises() -> None:
    with pytest.raises(RuntimeError, match="empty"):
        ContiguousLayerKV().read()


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        (
            torch.randn(2, 3, 4),
            torch.randn(2, 3, 4),
            "shape",
        ),
        (
            torch.randn(1, 2, 0, 4),
            torch.randn(1, 2, 0, 4),
            "greater than zero",
        ),
        (
            torch.randn(1, 2, 2, 4),
            torch.randn(1, 3, 2, 4),
            "identical shape",
        ),
        (
            torch.randn(1, 2, 2, 4),
            torch.randn(1, 2, 2, 4, dtype=torch.float64),
            "dtype",
        ),
    ],
)
def test_rejects_invalid_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ContiguousLayerKV().append(key, value)


def test_rejects_incompatible_append() -> None:
    cache = ContiguousLayerKV()
    key, value = make_kv(tokens=2)
    cache.append(key, value)

    with pytest.raises(ValueError, match="B and H"):
        cache.append(*make_kv(batch=2, tokens=1))

    with pytest.raises(ValueError, match="D cannot change"):
        cache.append(*make_kv(head_dim=8, tokens=1))

    with pytest.raises(ValueError, match="dtype"):
        cache.append(
            key.to(torch.float64),
            value.to(torch.float64),
        )
