import torch

from barellm.utils import check


class ContiguousLayerKV:
    """
    Expected tensor shapes:
        key/value: [B, num_kv_heads, T, head_dim]
    """

    def __init__(self) -> None:
        self.key: torch.Tensor | None = None
        self.value: torch.Tensor | None = None

    @property
    def seq_len(self) -> int:
        if self.key is None:
            return 0
        return self.key.shape[2]

    def append(
        self, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        check(key.ndim == 4 and value.ndim == 4, "K/V must have shape [B, H, T, D]")
        check(key.shape == value.shape, "K/V must have identical shape")
        check(key.shape[2] > 0, "K/V sequence length must be greater than zero")
        check(key.dtype == value.dtype, "K/V dtype must match")
        check(key.device == value.device, "K/V device must match")

        if self.key is None:
            self.key = key
            self.value = value
            return self.key, self.value

        if self.value is None:
            raise RuntimeError("key/value cache state is inconsistent")

        check(key.shape[:2] == self.key.shape[:2], "B and H cannot change")
        check(key.shape[3] == self.key.shape[3], "D cannot change")
        check(key.dtype == self.key.dtype, "K/V dtype cannot change")
        check(key.device == self.key.device, "K/V device cannot change")
        self.key = torch.cat([self.key, key], dim=2)
        self.value = torch.cat([self.value, value], dim=2)
        return self.key, self.value

    def read(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.key is None or self.value is None:
            raise RuntimeError("KV cache is empty")
        return self.key, self.value


class ContiguousKVCache:
    def __init__(self, num_layers: int) -> None:
        self.layers = [ContiguousLayerKV() for _ in range(num_layers)]

    def layer(self, layer_idx: int) -> ContiguousLayerKV:
        return self.layers[layer_idx]
