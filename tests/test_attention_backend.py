import pytest
import torch

from barellm.attention import create_attention_backend
from barellm.engine.batched_kv_cache import BatchKVCache
from barellm.engine.block_pool import BlockPool
from barellm.engine.contiguous_kv_cache import ContiguousKVCache
from barellm.engine.paged_kv_cache import PagedKVCache


def portable_devices() -> list[torch.device]:
    devices = [torch.device("cpu")]
    if torch.backends.mps.is_available():
        devices.append(torch.device("mps"))
    return devices


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def test_backend_selection_is_explicit() -> None:
    assert create_attention_backend("sdpa").__class__.__name__ == (
        "SDPAAttentionBackend"
    )
    assert create_attention_backend("auto").__class__.__name__ == (
        "SDPAAttentionBackend"
    )
    with pytest.raises(RuntimeError, match="not implemented"):
        create_attention_backend("triton")


@pytest.mark.parametrize("device", portable_devices(), ids=lambda item: str(item))
def test_sdpa_backend_runs_on_cpu_and_mps(device: torch.device) -> None:
    dtype = torch.float32
    storage = PagedKVCache(
        num_layers=1,
        max_blocks=2,
        num_kv_heads=2,
        block_size=2,
        head_dim=4,
        dtype=dtype,
        device=device,
    )
    pool = BlockPool(2)
    storage.register_request("request", pool.allocate(1))
    layer = storage.get_cache("request").layer(0)
    backend = create_attention_backend("sdpa")

    q = torch.randn(1, 4, 1, 4, device=device, dtype=dtype)
    key = torch.randn(1, 2, 1, 4, device=device, dtype=dtype)
    value = torch.randn(1, 2, 1, 4, device=device, dtype=dtype)

    output = backend.attend(q, key, value, layer, group_size=2)
    synchronize(device)

    assert output.shape == (1, 4, 1, 4)
    assert output.device.type == device.type
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("device", portable_devices(), ids=lambda item: str(item))
def test_batched_mask_stays_on_attention_device(device: torch.device) -> None:
    cache_a = ContiguousKVCache(num_layers=1)
    cache_b = ContiguousKVCache(num_layers=1)
    key_a = torch.randn(1, 2, 2, 3, device=device)
    value_a = torch.randn_like(key_a)
    key_b = torch.randn(1, 2, 3, 3, device=device)
    value_b = torch.randn_like(key_b)
    cache_a.layer(0).append(key_a, value_a)
    cache_b.layer(0).append(key_b, value_b)

    batch_layer = BatchKVCache([cache_a, cache_b]).layer(0)
    batch_layer.append(
        torch.randn(2, 2, 1, 3, device=device),
        torch.randn(2, 2, 1, 3, device=device),
    )
    mask = batch_layer.attention_mask(q_len=1)
    assert mask is not None
    synchronize(device)

    assert mask.device.type == device.type
    assert mask.shape == (2, 1, 1, 4)
