import torch
import torch.nn.functional as F
from torch import nn

from barellm.utils import check


class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        check(
            hidden_size % num_heads == 0,
            f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})",
        )
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads

        self.num_heads = num_heads

        self.q_proj = nn.Linear(
            self.hidden_size, self.head_dim * self.num_heads, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.head_dim * self.num_heads, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.head_dim * self.num_heads, bias=False
        )
        self.out_proj = nn.Linear(
            self.head_dim * self.num_heads, self.hidden_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape

        # [B, S, D] -> [B, S, H * D_h]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # [B, S, H * D_h] -> [B, S, H, D_h] -> [B, H, S, D_h]
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # SDPA expects [B, H, S, D_h]
        #     scores = Q @ K^T / sqrt(d) # [S, D_h] @ [D_h, S] -> [S, S]
        #     out = softmax(scores) @ V  # [S, S] @ [S, D_h] -> [S, D_h]
        context = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=True
        )

        # [B, H, S, D_h] -> [B, S, D]
        out = context.transpose(1, 2).reshape(B, S, D)
        return out


class GroupedQueryAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        check(
            hidden_size % num_heads == 0,
            f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})",
        )
        check(
            num_heads % num_kv_heads == 0,
            f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})",
        )

        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.group_size = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(
            self.hidden_size, self.head_dim * self.num_heads, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.head_dim * self.num_kv_heads, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.head_dim * self.num_kv_heads, bias=False
        )
        self.out_proj = nn.Linear(
            self.head_dim * self.num_heads, self.hidden_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape

        # [B, S, D] -> [B, S, H * D_h]
        q = self.q_proj(x)
        # [B, S, D] -> [B, S, H_kv * D_h]
        k = self.k_proj(x)
        v = self.v_proj(x)

        # [B, S, H * D_h] -> [B, S, H, D_h] -> [B, H, S, D_h]
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # [B, S, H_kv * D_h] -> [B, S, H_kv, D_h] -> [B, H_kv, S, D_h]
        k = k.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # [B, H_kv, S, D_h] -> [B, H, S, D_h]
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        # SDPA expects [B, H, S, D_h]
        context = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=True
        )

        # [B, H, S, D_h] -> [B, S, D]
        out = context.transpose(1, 2).reshape(B, S, D)
        return out
