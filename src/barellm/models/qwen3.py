import json
from dataclasses import dataclass, field, fields

import torch
from torch import nn

from barellm.hub import download_model
from barellm.models.block import TransformerBlock
from barellm.models.embedding import TiedLMHead, TokenEmbedding
from barellm.models.norm import RMSNorm
from barellm.models.weights import load_into_model, load_weights
from barellm.utils import check


@dataclass(frozen=True)
class Qwen3Config:
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    rope_scaling: dict | None
    tie_word_embeddings: bool
    bos_token_id: int
    eos_token_id: int
    hidden_act: str
    attention_bias: bool
    attention_dropout: float
    use_sliding_window: bool
    extras: dict = field(default_factory=dict)


KNOWN_EXTRAS: dict[str, str] = {
    "architectures": "HF runtime metadata",
    "initializer_range": "training-only; irrelevant for inference",
    "max_window_layers": "moot when use_sliding_window=False",
    "sliding_window": "moot when use_sliding_window=False",
    "torch_dtype": "we use barellm.config.DTYPE instead",
    "transformers_version": "HF library version stamp",
    "use_cache": "we manage KV cache ourselves",
}


def check_supported(cfg: Qwen3Config) -> None:
    check(cfg.hidden_act == "silu", f"unsupported {cfg.hidden_act}")
    check(cfg.attention_bias is False, "attention_bias not supported")
    check(cfg.attention_dropout == 0.0, "attention_dropout must be 0.0")
    check(cfg.use_sliding_window is False, "sliding window not supported")


def load_config(
    model_id: str = "Qwen/Qwen3-0.6B",
    warn_extras: bool = True,
) -> Qwen3Config:

    snapshot_dir = download_model(model_id)
    with open(snapshot_dir / "config.json") as f:
        raw = json.load(f)

    parsed_names = {f.name for f in fields(Qwen3Config)} - {"extras"}
    used = {k: raw[k] for k in parsed_names}
    extras = {k: v for k, v in raw.items() if k not in parsed_names}

    if warn_extras and extras:
        unknown = sorted(set(extras) - set(KNOWN_EXTRAS))
        print(
            f"[load_config] preserved {len(extras)} extra fields; "
            f"{len(unknown)} unrecognized: {unknown}"
        )
    cfg = Qwen3Config(extras=extras, **used)
    check_supported(cfg)
    return cfg


class Qwen3ForCausalLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        intermediate_size: int,
        head_dim: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        rope_theta: float = 1_000_000.0,
        use_qk_norm: bool = False,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.embed_tokens = TokenEmbedding(
            vocab_size=vocab_size, hidden_size=hidden_size
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=hidden_size,
                    head_dim=head_dim,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    intermediate_size=intermediate_size,
                    rms_norm_eps=rms_norm_eps,
                    rope_theta=rope_theta,
                    use_qk_norm=use_qk_norm,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(hidden_size, rms_norm_eps)
        self.lm_head = TiedLMHead(self.embed_tokens)

    @classmethod
    def from_config(cls, config: Qwen3Config) -> "Qwen3ForCausalLM":
        return cls(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            head_dim=config.head_dim,
            num_layers=config.num_hidden_layers,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=config.rope_theta,
            use_qk_norm=True,
            rms_norm_eps=config.rms_norm_eps,
        )

    def forward(
        self, token_ids: torch.Tensor, position_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        # [B, T] -> [B, T, D]
        h = self.embed_tokens(token_ids)
        for _layer_idx, layer in enumerate(self.layers):
            h = layer(h, position_ids=position_ids)
        # [B, T, D]
        h = self.norm(h)

        # [B, T, D] -> [B, T, vocab]
        return self.lm_head(h)


def load_qwen3(
    model_id: str = "Qwen/Qwen3-0.6B",
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Qwen3ForCausalLM:
    config = load_config(model_id)
    model = Qwen3ForCausalLM.from_config(config)
    model.to(device=device, dtype=dtype)

    weights = load_weights(
        model_id=model_id,
        device=device,
        dtype=dtype,
    )
    load_into_model(model, weights, dtype=dtype)

    return model.eval()
