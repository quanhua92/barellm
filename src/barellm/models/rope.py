import torch
from torch import nn

from barellm.utils import check


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float = 1_000_000):
        super().__init__()

        check(head_dim % 2 == 0, f"head_dim ({head_dim}) must be even for RoPE")

        self.head_dim = head_dim
        self.theta = theta

        exponents = (
            torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
        )  # [head_dim / 2]
        inv_freq = 1.0 / (theta**exponents)  # [head_dim / 2]

        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, H, T, D_h]
        # positions_ids: [B, T]
        B, _H, T, _D_h = x.shape

        if position_ids is None:
            position_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)

        positions = position_ids.to(device=x.device, dtype=self.inv_freq.dtype)

        # [B, T, 1] * [D_h/2] -> [B, T, D_h/2]
        angles = positions[..., None] * self.inv_freq

        # Half-split Qwen layout
        angles = torch.cat((angles, angles), dim=-1)

        # [B, T, D_h] -> [B, 1, T, D_h]
        cos = angles.cos().unsqueeze(1)
        sin = angles.sin().unsqueeze(1)

        return cos.to(x.dtype), sin.to(x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2

    first = x[..., :half]
    second = x[..., half:]

    return torch.cat((-second, first), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    x: [B, H, T, D_h]
    cos: [B, 1, T, D_h]
    """
    # [B, H, T, D_h]
    return x * cos + rotate_half(x) * sin
