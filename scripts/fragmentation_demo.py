# A pure Python script that simulates 10 concurrent LLM requests
# and compares two memory allocation strategies.
# No torch, no transformers, no model. Just math.
import random

random.seed(42)

MAX_CONTENT = 4096  # naive serving pre-allocated per request
BLOCK_SIZE = 16  # vLLM default

request_lengths = [random.randint(20, 3000) for _ in range(10)]
request_lengths.sort()

print("request_lengths=", request_lengths)

print("=" * 80)
print("Pre-allocated KV cache (naive)")

total_alloc = 0
total_used = 0

print(f"{'Request':<8} {'Actual':>8} {'Allocated':>10} {'Waste':>8} {'Waste %':>8}")
print("-" * 50)

for i, length in enumerate(request_lengths):
    allocated = MAX_CONTENT
    waste = allocated - length
    total_alloc += allocated
    total_used += length
    print(
        f"{i:^8} {length:>8} {allocated:>10} {waste:>8} {waste / allocated * 100:>7.1f}%"
    )

print("-" * 50)
overall_waste = (total_alloc - total_used) / total_alloc * 100
print(
    f"{'Total':<8} {total_used:>8} {total_alloc:>10} {total_alloc - total_used:>8} {overall_waste:>7.1f}%"
)


print("=" * 80)
print(f"Paged allocation (block_size={BLOCK_SIZE})")

print(
    f"{'Request':<8} {'Actual':>8} {'Blocks':>8} {'Allocated':>10} {'Waste':>8} {'Waste %':>8}"
)
print("-" * 50)

paged_alloc = 0
paged_used = 0
block_used = 0

for i, length in enumerate(request_lengths):
    n_blocks = (length + BLOCK_SIZE - 1) // BLOCK_SIZE
    allocated = n_blocks * BLOCK_SIZE
    waste = allocated - length
    paged_alloc += allocated
    paged_used += length
    block_used += n_blocks
    print(
        f"{i:^8} {length:>8} {n_blocks:>8} {allocated:>10} {waste:>8} {waste / allocated * 100:>7.1f}%"
    )

print("-" * 50)
paged_waste = (paged_alloc - paged_used) / paged_alloc * 100
print(
    f"{'Total':<8} {paged_used:>8} {block_used:>8} {paged_alloc:>10} {paged_alloc - paged_used:>8} {paged_waste:>7.1f}%"
)

print("=" * 80)
print(f"Pre-allocated waste: {overall_waste:.1f}%")
print(f"Paged waste:         {paged_waste:.1f}%")
print(f"Memory saved:        {(1 - paged_alloc / total_alloc) * 100:.0f}%")
