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

    @property
    def seq_len(self) -> int:
        lengths = [layer.seq_len for layer in self.layers]
        check(
            len(set(lengths)) == 1,
            "equal cache lengths required for initial batching",
        )
        return lengths[0]

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
        lengths = [layer.seq_len for layer in self.layers]
        check(
            len(set(lengths)) == 1,
            "equal cache lengths required for initial batching",
        )

        keys = []
        values = []
        for batch_idx, layer in enumerate(self.layers):
            cached_key, cached_value = layer.append(
                key[batch_idx : batch_idx + 1],
                value[batch_idx : batch_idx + 1],
            )
            keys.append(cached_key)
            values.append(cached_value)

        check(
            len({item.shape[2] for item in keys}) == 1,
            "equal cache lengths required for initial batching",
        )
        return torch.cat(keys, dim=0), torch.cat(values, dim=0)

    def read(self) -> tuple[torch.Tensor, torch.Tensor]:
        keys = []
        values = []
        for layer in self.layers:
            key, value = layer.read()
            keys.append(key)
            values.append(value)

        check(
            len({item.shape[2] for item in keys}) == 1,
            "equal cache lengths required for initial batching",
        )
        return torch.cat(keys, dim=0), torch.cat(values, dim=0)
