from dataclasses import dataclass


@dataclass(slots=True)
class KVCacheBlock:
    block_id: int


class BlockPool:
    def __init__(self, max_blocks):
        self.blocks = [KVCacheBlock(i) for i in range(max_blocks)]
        self.free_ids = list(range(max_blocks))

    def can_allocate(self, num_blocks: int) -> bool:
        return len(self.free_ids) >= num_blocks

    def allocate(self, num_blocks: int) -> list[KVCacheBlock]:
        if num_blocks > len(self.free_ids):
            return []
        result = []
        for _ in range(num_blocks):
            block_id = self.free_ids.pop()
            result.append(self.blocks[block_id])
        return result

    def free(self, blocks: list[KVCacheBlock]) -> None:
        for block in blocks:
            self.free_ids.append(block.block_id)
