import torch
import torch.nn.functional as F
from torch import nn

from barellm.models.cache import KVCache
from barellm.models.norm import RMSNorm
from barellm.models.rope import RotaryEmbedding, apply_rope
from barellm.utils import check


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_heads: int,
        rope_theta: float = 1_000_000.0,
        use_qk_norm: bool = False,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        check(hidden_size > 0, "hidden_size must be positive")
        check(num_heads > 0, "num_heads must be positive")
        check(head_dim > 0 and head_dim % 2 == 0, "head_dim must be positive and even")

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_heads = num_heads

        self.q_proj = nn.Linear(hidden_size, head_dim * num_heads, bias=False)
        self.k_proj = nn.Linear(hidden_size, head_dim * num_heads, bias=False)
        self.v_proj = nn.Linear(hidden_size, head_dim * num_heads, bias=False)
        self.o_proj = nn.Linear(head_dim * num_heads, hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(head_dim, rope_theta)
        self.use_qk_norm = use_qk_norm
        if self.use_qk_norm:
            self.q_norm = RMSNorm(head_dim, rms_norm_eps)
            self.k_norm = RMSNorm(head_dim, rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        layer_idx: int = 0,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        B, T, _D = x.shape

        # [B, T, _D] -> [B, T, H * D_h]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # [B, T, H * D_h] -> [B, T, H, D_h] -> [B, H, T, D_h]
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # QK Norm before RoPE
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # [B, 1, T, D]
        cos, sin = self.rotary_emb(q, position_ids)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if kv_cache is not None:
            layer_cache = kv_cache.layer(layer_idx)
            k, v = layer_cache.append(k, v)

        q_len = q.shape[2]
        is_causal = q_len > 1
        context = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

        # [B, H, T, D_h] -> [B, T, D]
        context = context.transpose(1, 2).reshape(B, T, -1)

        return self.o_proj(context)


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 1_000_000.0,
        use_qk_norm: bool = False,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        check(hidden_size > 0, "hidden_size must be positive")
        check(num_heads > 0, "num_heads must be positive")
        check(num_kv_heads > 0, "num_kv_heads must be positive")
        check(head_dim > 0 and head_dim % 2 == 0, "head_dim must be positive and even")
        check(
            num_heads % num_kv_heads == 0,
            "num_heads must divide evenly by num_kv_heads",
        )

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.group_size = num_heads // num_kv_heads

        self.q_proj = nn.Linear(hidden_size, head_dim * num_heads, bias=False)
        self.k_proj = nn.Linear(hidden_size, head_dim * num_kv_heads, bias=False)
        self.v_proj = nn.Linear(hidden_size, head_dim * num_kv_heads, bias=False)
        self.o_proj = nn.Linear(head_dim * num_heads, hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(head_dim, rope_theta)
        self.use_qk_norm = use_qk_norm
        if self.use_qk_norm:
            self.q_norm = RMSNorm(head_dim, rms_norm_eps)
            self.k_norm = RMSNorm(head_dim, rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        layer_idx: int = 0,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        B, T, _D = x.shape

        # [B, T, _D] -> [B, T, H * D_h]
        q = self.q_proj(x)
        # [B, T, _D] -> [B, T, H_kv * D_h]
        k = self.k_proj(x)
        v = self.v_proj(x)

        # [B, T, H * D_h] -> [B, T, H, D_h] -> [B, H, T, D_h]
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        # [B, T, H_kv * D_h] -> [B, T, H_kv, D_h] -> [B, H_kv, T, D_h]
        k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # QK Norm before RoPE
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # [B, 1, T, D]
        cos, sin = self.rotary_emb(q, position_ids)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if kv_cache is not None:
            layer_cache = kv_cache.layer(layer_idx)
            k, v = layer_cache.append(k, v)

        # [B, H_kv, T, D_h] -> [B, H, T, D_h]
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        q_len = q.shape[2]
        is_causal = q_len > 1
        context = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

        # [B, H, T, D_h] -> [B, T, D]
        context = context.transpose(1, 2).reshape(B, T, -1)

        return self.o_proj(context)
