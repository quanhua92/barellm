from typing import Protocol

import torch


class LayerKV(Protocol):
    """
    Expected tensor shapes:
        key/value: [B, num_kv_heads, T, head_dim]
    """

    @property
    def seq_len(self) -> int: ...

    def append(
        self, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def read(self) -> tuple[torch.Tensor, torch.Tensor]: ...


class KVCache(Protocol):
    def layer(self, layer_idx: int) -> LayerKV: ...
