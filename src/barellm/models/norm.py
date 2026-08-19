import torch
from torch import nn

from barellm.utils import check


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        check(dim > 0, f"dim ({dim}) must be > 0")
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype

        x_float = x.float()
        mean_square = x_float.pow(2).mean(dim=-1, keepdim=True)

        normalized = x_float * torch.rsqrt(mean_square + self.eps)

        return (normalized * self.weight).to(input_dtype)
