NUM_PHYSICAL_BLOCKS = 20
BLOCK_SIZE = 16

free_blocks = list(range(NUM_PHYSICAL_BLOCKS))
print(f"free_blocks: \n{free_blocks}")

# Allocate Request A
block_table_a = []
for _ in range(3):
    physical = free_blocks.pop(0)
    block_table_a.append(physical)

print("===Allocate Request A===")
print(f"block_table_a: \n{block_table_a}")
print(f"free_blocks: \n{free_blocks}")


# Allocate Request B with 3 blocks and shares block 0 with A
block_table_b = [block_table_a[0]]
for _ in range(2):
    physical = free_blocks.pop(0)
    block_table_b.append(physical)

print("===Allocate Request B - shares block 0 with A===")
print(f"block_table_b: \n{block_table_b}")
print(f"free_blocks: \n{free_blocks}")

print("Physical block layout")
for p in range(NUM_PHYSICAL_BLOCKS):
    owners = []
    if p in block_table_a:
        owners.append("A")
    if p in block_table_b:
        owners.append("B")
    status = f"owners: {','.join(owners)}" if owners else "FREE"
    print(f"   physical[{p:>2}]: {status}")

print("How attention reads K,V for request A, token at position 35:")
logical_block = 35 // BLOCK_SIZE  # = 2
offset = 35 % BLOCK_SIZE  # =3
physical_block = block_table_a[logical_block]  # = 2
print(f"   logical block = 35 // {BLOCK_SIZE}, offset = 35 % {BLOCK_SIZE}")
print(f"   logical block {logical_block}, offset {offset}")
print(f"    -> physical block {physical_block}")
print(f"    -> key_cache[physical[{physical_block}]][{offset}]")
