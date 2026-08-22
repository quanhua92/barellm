from dataclasses import dataclass

import torch

from barellm.engine.block_pool import KVCacheBlock
from barellm.models.cache import KVCache, LayerKV
from barellm.utils import check


@dataclass
class _RequestPages:
    blocks: list[KVCacheBlock]
    block_table: torch.Tensor
    layer_seq_lens: list[int]


class PagedKVCache:
    """Paged K/V storage with dense gathering for the current SDPA backend."""

    def __init__(
        self,
        num_layers: int,
        max_blocks: int,
        num_kv_heads: int,
        block_size: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        check(num_layers > 0, "num_layers must be positive")
        check(max_blocks > 0, "max_blocks must be positive")
        check(num_kv_heads > 0, "num_kv_heads must be positive")
        check(block_size > 0, "block_size must be positive")
        check(head_dim > 0, "head_dim must be positive")

        self.num_layers = num_layers
        self.max_blocks = max_blocks
        self.num_kv_heads = num_kv_heads
        self.block_size = block_size
        self.head_dim = head_dim
        self.dtype = dtype
        shape = (
            num_layers,
            max_blocks,
            num_kv_heads,
            block_size,
            head_dim,
        )
        self.key_cache = torch.empty(shape, dtype=dtype, device=device)
        self.value_cache = torch.empty_like(self.key_cache)
        self.device = self.key_cache.device
        self.request_pages: dict[str, _RequestPages] = {}

    def register_request(
        self,
        request_id: str,
        blocks: list[KVCacheBlock],
    ) -> None:
        if request_id not in self.request_pages:
            self.request_pages[request_id] = _RequestPages(
                blocks=list(blocks),
                block_table=torch.tensor(
                    [block.block_id for block in blocks],
                    dtype=torch.long,
                    device=self.device,
                ),
                layer_seq_lens=[0] * self.num_layers,
            )
        else:
            self.request_pages[request_id].blocks = list(blocks)
            self.request_pages[request_id].block_table = torch.tensor(
                [block.block_id for block in blocks],
                dtype=torch.long,
                device=self.device,
            )

    def unregister_request(self, request_id: str) -> None:
        self.request_pages.pop(request_id, None)

    def get_cache(self, request_id: str) -> KVCache:
        if request_id not in self.request_pages:
            raise KeyError(f"Unknown request: {request_id}")
        return PagedKVCacheView(self, request_id)


class PagedKVCacheView:
    def __init__(self, storage: PagedKVCache, request_id: str) -> None:
        self.storage = storage
        self.request_id = request_id

    def layer(self, layer_idx: int) -> LayerKV:
        if not 0 <= layer_idx < self.storage.num_layers:
            raise IndexError(f"Invalid layer index: {layer_idx}")
        return PagedLayerKV(self.storage, self.request_id, layer_idx)


class PagedLayerKV:
    def __init__(
        self,
        storage: PagedKVCache,
        request_id: str,
        layer_idx: int,
    ) -> None:
        self.storage = storage
        self.request_id = request_id
        self.layer_idx = layer_idx

    @property
    def _request(self) -> _RequestPages:
        try:
            return self.storage.request_pages[self.request_id]
        except KeyError as exc:
            raise KeyError(f"Unknown request: {self.request_id}") from exc

    @property
    def seq_len(self) -> int:
        return self._request.layer_seq_lens[self.layer_idx]

    def append(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        check(key.ndim == 4 and value.ndim == 4, "K/V must have shape [B, H, T, D]")
        check(key.shape == value.shape, "K/V must have identical shape")
        check(key.shape[0] == 1, "PagedKVCache currently supports batch size 1")
        check(key.shape[1] == self.storage.num_kv_heads, "KV-head count cannot change")
        check(key.shape[2] > 0, "K/V sequence length must be greater than zero")
        check(key.shape[3] == self.storage.head_dim, "Head dimension cannot change")
        check(key.dtype == self.storage.dtype, "K/V dtype cannot change")
        check(key.device == self.storage.device, "K/V device cannot change")
        check(key.dtype == value.dtype, "K/V dtype must match")
        check(key.device == value.device, "K/V device must match")

        start = self.seq_len
        end = start + key.shape[2]

        required_blocks = (end + self.storage.block_size - 1) // self.storage.block_size
        check(
            required_blocks <= len(self._request.blocks),
            "request does not have enough allocated KV blocks",
        )

        positions = torch.arange(
            start, end, device=self.storage.device, dtype=torch.long
        )
        logical_blocks = positions // self.storage.block_size  # [T]
        block_offsets = positions % self.storage.block_size  # [T]
        physical_blocks = self._request.block_table[logical_blocks]  # [T]

        key_values = key[0].transpose(0, 1)  # [T, H, D]
        value_values = value[0].transpose(0, 1)  # [T, H, D]

        self.storage.key_cache[self.layer_idx, physical_blocks, :, block_offsets, :] = (
            key_values
        )
        self.storage.value_cache[
            self.layer_idx, physical_blocks, :, block_offsets, :
        ] = value_values

        self._request.layer_seq_lens[self.layer_idx] = end
        return self.read()

    def read(self) -> tuple[torch.Tensor, torch.Tensor]:
        check(self.seq_len > 0, "KV cache is empty")

        positions = torch.arange(
            self.seq_len, device=self.storage.device, dtype=torch.long
        )
        logical_blocks = positions // self.storage.block_size  # [T]
        block_offsets = positions % self.storage.block_size  # [T]
        physical_blocks = self._request.block_table[logical_blocks]  # [T]

        # self.storage.key_cache: [L, P, H, S, D] = [layer, physical_block, kv_head, offset_inside_block, head_dim]
        # keys: [T, H, D]
        # values: [T, H, D]
        # Pytorch advanced indexing: tensor[integer, index_tensor, :, index_tensor, :]
        keys = self.storage.key_cache[
            self.layer_idx, physical_blocks, :, block_offsets, :
        ]
        values = self.storage.value_cache[
            self.layer_idx, physical_blocks, :, block_offsets, :
        ]

        # [T, H, D] -> [B, H, T, D]
        return (
            keys.permute(1, 0, 2).unsqueeze(0),
            values.permute(1, 0, 2).unsqueeze(0),
        )

    def attention_mask(self, q_len: int) -> torch.Tensor | None:
        del q_len
        return None
