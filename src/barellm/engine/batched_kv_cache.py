import torch

from barellm.models.cache import KVCache, LayerKV
from barellm.utils import check


class BatchKVCache:
    def __init__(self, caches: list[KVCache]) -> None:
        check(len(caches) > 0, "batch cache cannot be empty")
        self.caches = caches

    def layer(self, layer_idx: int) -> LayerKV:
        return BatchLayerKV([cache.layer(layer_idx) for cache in self.caches])


class BatchLayerKV:
    def __init__(self, layers: list[LayerKV]) -> None:
        self.layers = layers
        self.lengths = torch.tensor(
            [layer.seq_len for layer in layers],
            dtype=torch.long,
        )

    @property
    def seq_len(self) -> int:
        return int(self.lengths.max().item())

    def append(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        check(
            key.shape[0] == len(self.layers),
            "batch size must match number of caches",
        )
        check(
            value.shape[0] == len(self.layers),
            "batch size must match number of caches",
        )
        keys = []
        values = []
        lengths = []
        for batch_idx, layer in enumerate(self.layers):
            cached_key, cached_value = layer.append(
                key[batch_idx : batch_idx + 1],
                value[batch_idx : batch_idx + 1],
            )
            keys.append(cached_key)
            values.append(cached_value)
            lengths.append(cached_key.shape[2])

        self.lengths = torch.tensor(lengths, device=key.device)
        return self._pad(keys), self._pad(values)

    def read(self) -> tuple[torch.Tensor, torch.Tensor]:
        keys = []
        values = []
        for layer in self.layers:
            key, value = layer.read()
            keys.append(key)
            values.append(value)

        self.lengths = torch.tensor(
            [item.shape[2] for item in keys],
            dtype=torch.long,
            device=keys[0].device,
        )
        return self._pad(keys), self._pad(values)

    def _pad(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        max_len = max(tensor.shape[2] for tensor in tensors)
        first = tensors[0]
        result = torch.zeros(
            len(tensors),
            first.shape[1],
            max_len,
            first.shape[3],
            dtype=first.dtype,
            device=first.device,
        )

        for batch_idx, tensor in enumerate(tensors):
            result[batch_idx, :, : tensor.shape[2], :] = tensor[0]

        return result

    def attention_mask(self, q_len: int) -> torch.Tensor:
        max_len = int(self.lengths.max().item())
        valid_keys = torch.arange(max_len, device=self.lengths.device).unsqueeze(
            0
        ) < self.lengths.unsqueeze(1)
        return valid_keys[:, None, None, :].expand(-1, 1, q_len, -1)
