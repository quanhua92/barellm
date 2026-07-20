import math

import torch

theta = 1_000_000  # Qwen3
head_dim = 128
max_seq = 256

i = torch.arange(head_dim // 2)  # [0, 1, 2, 3, ..., 63]
freqs = theta ** (-2 * i / head_dim)  # [64]
print(f"freqs = {freqs.shape} [64]")
print(freqs)

print("\n=== Plane periods ===")
print(
    "How many positions until each plane completes on full rotation. Notice plane 63 barely move"
)
for idx in [0, 1, 32, 63]:
    period = 2 * math.pi / freqs[idx]
    print(f"    plane {idx:>2}: freq={freqs[idx]:.6f} period={period:.0f} positions")

print("\n=== Rotation angle per position x plane ===")
t = torch.arange(max_seq).float()
freqs_matrix = torch.outer(t, freqs)  # [256, 64]
print(f"freqs_matrix = angle table {freqs_matrix.shape} [256, 64]")
print(f"    Row = position (0 to {max_seq - 1})")
print(f"    Column = plane (0=fast to {head_dim // 2 - 1}=slow)")
print(freqs_matrix)
cos = freqs_matrix.cos()
sin = freqs_matrix.sin()
print(f"cos = {cos.shape} [256, 64]")
print(f"sin = {sin.shape} [256, 64]")


def apply_rope_split(x, cos_row, sin_row):
    half = x.shape[-1] // 2
    x1 = x[:half]  # dim 0-63
    x2 = x[half:]  # dim 64-127
    c = cos_row[:half]
    s = sin_row[:half]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c])


# random Q, K vectors
torch.manual_seed(42)
q = torch.randn(head_dim)
k = torch.randn(head_dim)

dot_products = torch.zeros(max_seq, max_seq)

for m in range(max_seq):
    q_rot = apply_rope_split(q, cos[m], sin[m])
    for n in range(max_seq):
        k_rot = apply_rope_split(k, cos[n], sin[n])
        dot_products[m, n] = torch.dot(q_rot, k_rot)

print("\n=== Relative position property ===")
print(
    "If RoPE works, Q_m . K_n depends only on (m - n), not absolute positions. So, these should give same values"
)
print(f"dot_products[5, 3]     = {dot_products[5, 3]:.6f}  (m=5,  n=3 -> m - n = 2)")
print(f"dot_products[10, 8]    = {dot_products[10, 8]:.6f}  (m=10,  n=8 -> m - n = 2)")
print(
    f"dot_products[20, 18]   = {dot_products[20, 18]:.6f}  (m=20,  n=18 -> m - n = 2)"
)
match = torch.allclose(
    dot_products[5, 3], dot_products[10, 8], atol=1e-4
) and torch.allclose(dot_products[10, 8], dot_products[20, 18], atol=1e-4)
print(f"All same? {match}")

print("\n=== Dot Products matrix (diagonal constancy)===")
print(
    "Each row is one offset. std~0 means all pairs at that offset have the same value = relative positions confirmed"
)
print(
    "offset 0 = same position (highest value). offset + k != offset - k because direction matters"
)
print(f"{'offset':>8} {'mean':>10} {'std':>12}")
for offset in range(-5, 6):
    vals = [
        dot_products[i, i + offset]
        for i in range(max(0, -offset), min(max_seq, max_seq - offset))
    ]
    mean = sum(vals) / len(vals)
    std = torch.tensor(vals).std()
    print(f"{offset:>+8d} {mean:>10.4f} {std:>12.6f}")

print("\n=== theta comparison at position 100 ===")
print(
    "theta controls how planes rotate. Bigger theta = slower = longer context before planes wrap"
)

plane_idx = 15
for theta in [1_000, 10_000, 100_000, 1_000_000, 10_000_000]:
    i = torch.arange(head_dim // 2)  # [0, 1, 2, 3, ..., 63]
    freqs = theta ** (-2 * i / head_dim)  # [64]
    angle = 100 * freqs[plane_idx]  # plane 15 at position 100
    rotations = angle / (2 * math.pi)
    print(
        f"  theta={theta:>10,}: plane {plane_idx} has rotated {rotations:.1f} times at pos 100"
    )
