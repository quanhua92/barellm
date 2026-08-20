from collections.abc import Callable

import torch
from safetensors import safe_open
from torch import nn

from barellm.hub import download_model


def default_map_key(key: str) -> str:
    """Default weight name mapping: strip the HF 'model.' prefix.

    Module names already match HF (embed_tokens, self_attn, o_proj,
    input_layernorm, post_attention_layernorm), so only the prefix needs
    stripping. Pass a custom function for models with different naming.
    """
    return key.removeprefix("model.")


def load_weights(
    model_id: str, device: str, dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    snapshot_dir = download_model(model_id)
    safetensors_files = sorted(snapshot_dir.glob("*.safetensors"))
    if not safetensors_files:
        raise FileNotFoundError(f"No safetensors files in {snapshot_dir}")
    weights: dict[str, torch.Tensor] = {}

    for path in safetensors_files:
        with safe_open(str(path), framework="pt", device=device) as f:
            for key in f.keys():  # noqa: SIM118 - safe_open is not a dict
                if key in weights:
                    raise ValueError(f"Duplicate tensor key {key} across shards")
                weights[key] = f.get_tensor(key).to(dtype)
    return weights


def load_into_model(
    model: nn.Module,
    weights: dict[str, torch.Tensor],
    dtype: torch.dtype | None = None,
    map_key: Callable[[str], str] = default_map_key,
) -> None:
    """Load HF checkpoint weights into model.

    Args:
        model: the target nn.Module.
        weights: raw HF state dict (from load_weights).
        map_key: function translating HF tensor names to model param names.
            Defaults to stripping the 'model.' prefix.

    Note: load_state_dict upcasts to fp32, so we cast back to DTYPE
    afterwards — otherwise the model silently runs fp32.
    """
    sd = {map_key(k): v for k, v in weights.items()}
    model.load_state_dict(sd, strict=True)
    if dtype is not None:
        model.to(dtype=dtype)
