import torch
from torch import nn

from barellm.utils import check


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq: int = 2048, theta: float = 1000000.0):
        super().__init__()

        check(head_dim % 2 == 0, f"head_dim ({head_dim}) must be even for RoPE")

        self.head_dim = head_dim
        self.max_seq = max_seq
        self.theta = theta

        # Compute inverse freq in fp32 for numerical stability
        exponents = (
            torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
        )  # [head_dim / 2]
        inv_freq = 1.0 / (theta**exponents)  # [head_dim / 2]
        # inv_freq[i] is the angular speed of frequency plane i
        # At position m, the plane has rotated by m * inv_freq[i] radians
        positions = torch.arange(max_seq, dtype=torch.float32)  # [max_seq]
        angles = (
            positions[:, None] * inv_freq[None, :]
        )  # [max_seq, head_dim / 2] - outer product via broadcasting
        # angles[m, i] = how many radians frequency plane i has rotated at position m

        # Duplicate: each frequency controls a pair (head_dim i, head_dim i+head_dim/2)
        # Both dims use the same angle, so emb = [a0..a{d/2-1}, a0..a{d/2-1}]
        emb = torch.cat((angles, angles), dim=-1)  # [max_seq, head_dim]

        self.register_buffer("cos_cache", emb.cos(), persistent=False)
        self.register_buffer("sin_cache", emb.sin(), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids is None:
            B, _H, S, _D_h = x.shape
            position_ids = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)
        # position_ids: [B, S] -> cos/sin: [B, 1, S, D_h]
        cos = self.cos_cache[position_ids].unsqueeze(1).to(dtype=x.dtype)
        sin = self.sin_cache[position_ids].unsqueeze(1).to(dtype=x.dtype)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Split x in half along the last dim, negate-and-swap.

    [..., d] -> [-x[..., d/2:], x[..., :d/2]]

    This is the split-half pairing HF uses for RoPE.
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to x at given positions.

    Args:
        q: [B, H_q, S, D_h].
        k: [B, H_k, S, D_h].
        cos: [B, 1, S, D_h].
        sin: [B, 1, S, D_h].

    Returns:
        Tuple of (q_embed, k_embed), each same shape as inputs
    """
    q_embed = q * cos + (rotate_half(q) * sin)
    k_embed = k * cos + (rotate_half(k) * sin)

    return q_embed, k_embed
