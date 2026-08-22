import torch
from torch import nn

from barellm.attention import AttentionBackendName
from barellm.models.attention import CausalSelfAttention, GroupedQueryAttention
from barellm.models.cache import KVCache
from barellm.models.mlp import SwiGLU
from barellm.models.norm import RMSNorm
from barellm.utils import check


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_heads: int,
        intermediate_size: int,
        num_kv_heads: int | None = None,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1_000_000.0,
        use_qk_norm: bool = False,
        attention_backend: AttentionBackendName = "sdpa",
    ):
        super().__init__()
        check(hidden_size > 0, "hidden_size must be positive")
        check(num_heads > 0, "num_heads must be positive")
        check(head_dim > 0 and head_dim % 2 == 0, "head_dim must be positive and even")
        check(intermediate_size > 0, "intermediate_size must be positive")

        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        if num_kv_heads is not None:
            self.self_attn = GroupedQueryAttention(
                hidden_size=hidden_size,
                head_dim=head_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                rope_theta=rope_theta,
                use_qk_norm=use_qk_norm,
                rms_norm_eps=rms_norm_eps,
                attention_backend=attention_backend,
            )
        else:
            self.self_attn = CausalSelfAttention(
                hidden_size=hidden_size,
                head_dim=head_dim,
                num_heads=num_heads,
                rope_theta=rope_theta,
                use_qk_norm=use_qk_norm,
                rms_norm_eps=rms_norm_eps,
                attention_backend=attention_backend,
            )

        self.mlp = SwiGLU(hidden_size=hidden_size, intermediate_size=intermediate_size)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        layer_idx: int = 0,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        residual = x

        x = self.input_layernorm(x)
        x = self.self_attn(
            x, layer_idx=layer_idx, position_ids=position_ids, kv_cache=kv_cache
        )
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)

        return residual + x
