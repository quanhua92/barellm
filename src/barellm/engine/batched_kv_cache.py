import torch

from barellm.engine.paged_kv_cache import (
    PagedBatchKVMetadata,
    PagedLayerKV,
)
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
        return max(layer.seq_len for layer in self.layers)

    def append(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.append_only(key, value)
        return self.read()

    def append_only(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        check(
            key.shape[0] == len(self.layers),
            "batch size must match number of caches",
        )
        check(
            value.shape[0] == len(self.layers),
            "batch size must match number of caches",
        )
        for batch_idx, layer in enumerate(self.layers):
            layer.append_only(
                key[batch_idx : batch_idx + 1],
                value[batch_idx : batch_idx + 1],
            )

        self.lengths = torch.tensor(
            [layer.seq_len for layer in self.layers],
            dtype=torch.long,
            device=key.device,
        )

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
        max_len = max(layer.seq_len for layer in self.layers)
        valid_keys = torch.arange(max_len, device=self.lengths.device).unsqueeze(
            0
        ) < self.lengths.unsqueeze(1)
        return valid_keys[:, None, None, :].expand(-1, 1, q_len, -1)

    def paged_metadata_post_append(self) -> PagedBatchKVMetadata:
        """Build checked rectangular metadata for homogeneous paged layers."""
        if not all(isinstance(layer, PagedLayerKV) for layer in self.layers):
            raise TypeError("paged metadata requires PagedLayerKV members")

        paged_layers = [
            layer for layer in self.layers if isinstance(layer, PagedLayerKV)
        ]
        metadata = [layer.paged_metadata_post_append() for layer in paged_layers]
        first = metadata[0]
        if any(
            item.key_cache is not first.key_cache
            or item.value_cache is not first.value_cache
            or item.layer_idx != first.layer_idx
            or item.block_size != first.block_size
            for item in metadata[1:]
        ):
            raise ValueError("paged batch layers must share storage and layout")

        max_blocks = max(item.block_table.numel() for item in metadata)
        block_tables = torch.full(
            (len(metadata), max_blocks),
            -1,
            dtype=torch.int32,
            device=first.block_table.device,
        )
        for row, item in enumerate(metadata):
            block_tables[row, : item.block_table.numel()] = item.block_table

        seq_lens = torch.tensor(
            [item.seq_len for item in metadata],
            dtype=torch.int32,
            device=first.block_table.device,
        )
        return PagedBatchKVMetadata(
            key_cache=first.key_cache,
            value_cache=first.value_cache,
            layer_idx=first.layer_idx,
            block_tables=block_tables,
            seq_lens=seq_lens,
            block_size=first.block_size,
        )
