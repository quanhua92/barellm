from collections.abc import Callable
from typing import Literal, Protocol

import torch
import torch.nn.functional as F

from barellm.models.cache import LayerKV

AttentionBackendName = Literal["auto", "sdpa", "triton"]


class AttentionBackend(Protocol):
    """Backend contract for append-plus-attend operations."""

    def attend(
        self,
        q: torch.Tensor,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        layer_cache: LayerKV | None,
        group_size: int,
    ) -> torch.Tensor: ...


class SDPAAttentionBackend:
    """Portable reference backend using dense K/V and PyTorch SDPA."""

    def __init__(
        self,
        sdpa: Callable[..., torch.Tensor] | None = None,
    ) -> None:
        self._sdpa = sdpa or F.scaled_dot_product_attention

    def attend(
        self,
        q: torch.Tensor,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        layer_cache: LayerKV | None,
        group_size: int,
    ) -> torch.Tensor:
        if layer_cache is None:
            key, value = new_k, new_v
        else:
            key, value = layer_cache.append(new_k, new_v)

        key = key.repeat_interleave(group_size, dim=1)
        value = value.repeat_interleave(group_size, dim=1)
        q_len = q.shape[2]
        attn_mask = None if layer_cache is None else layer_cache.attention_mask(q_len)

        # A one-token cached decode has past keys plus the current key. PyTorch's
        # non-square causal mask is upper-left aligned, so is_causal=True would
        # incorrectly hide the current token in that case.
        return self._sdpa(
            q,
            key,
            value,
            attn_mask=attn_mask,
            is_causal=attn_mask is None and q_len > 1,
        )


def create_attention_backend(
    name: AttentionBackendName,
    *,
    sdpa: Callable[..., torch.Tensor] | None = None,
) -> AttentionBackend:
    """Create a backend without importing optional Triton dependencies."""
    if name not in ("auto", "sdpa", "triton"):
        raise ValueError(f"Unknown attention backend: {name!r}")
    if name == "triton":
        raise RuntimeError(
            "The Triton attention backend is not implemented yet; "
            "use attention_backend='sdpa'"
        )
    # `auto` intentionally resolves to the portable path until a Triton
    # implementation and capability checks are available.
    return SDPAAttentionBackend(sdpa=sdpa)
