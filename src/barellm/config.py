"""Environment-backed BareLLM settings.

Real environment variables take precedence over values loaded from ``.env``.
The legacy ``MODEL_ID``, ``DEVICE``, and ``DTYPE`` names remain available for
the existing model and example code.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from dotenv import load_dotenv

DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
DEFAULT_PROFILE_ROOT = Path("profiles")


@dataclass(frozen=True)
class Settings:
    model_id: str
    device: str
    dtype: torch.dtype
    host: str
    port: int
    profile_root: Path
    enable_profile_api: bool


def _get(environment: Mapping[str, str], key: str, default: str) -> str:
    return environment.get(key, default)


def _parse_bool(value: str, key: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{key} must be a boolean value")


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_device(value: str) -> str:
    normalized = value.strip().lower()
    if normalized == "auto":
        return _detect_device()
    if normalized not in {"cpu", "cuda", "mps"}:
        raise ValueError("BARELLM_DEVICE must be auto, cpu, cuda, or mps")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise ValueError("BARELLM_DEVICE=cuda but CUDA is unavailable")
    if normalized == "mps" and not torch.backends.mps.is_available():
        raise ValueError("BARELLM_DEVICE=mps but MPS is unavailable")
    return normalized


def _resolve_dtype(value: str, device: str) -> torch.dtype:
    normalized = value.strip().lower()
    if normalized == "auto":
        if device == "cuda":
            return torch.bfloat16
        if device == "mps":
            return torch.float16
        return torch.float32
    dtypes = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in dtypes:
        raise ValueError("BARELLM_DTYPE must be auto, float32, float16, or bfloat16")
    return dtypes[normalized]


def load_settings(
    *,
    environment: Mapping[str, str] | None = None,
    load_dotenv_file: bool = True,
) -> Settings:
    """Load settings from environment variables and an optional ``.env``."""
    if environment is None:
        if load_dotenv_file:
            load_dotenv(override=False)
        environment = os.environ

    device = _resolve_device(_get(environment, "BARELLM_DEVICE", "auto"))
    port_text = _get(environment, "BARELLM_PORT", str(DEFAULT_PORT))
    try:
        port = int(port_text)
    except ValueError as exc:
        raise ValueError("BARELLM_PORT must be an integer") from exc
    if not 1 <= port <= 65535:
        raise ValueError("BARELLM_PORT must be between 1 and 65535")

    return Settings(
        model_id=_get(environment, "BARELLM_MODEL_ID", DEFAULT_MODEL_ID),
        device=device,
        dtype=_resolve_dtype(
            _get(environment, "BARELLM_DTYPE", "auto"),
            device,
        ),
        host=_get(environment, "BARELLM_HOST", DEFAULT_HOST),
        port=port,
        profile_root=Path(
            _get(
                environment,
                "BARELLM_PROFILE_ROOT",
                str(DEFAULT_PROFILE_ROOT),
            )
        ),
        enable_profile_api=_parse_bool(
            _get(environment, "BARELLM_ENABLE_PROFILE_API", "true"),
            "BARELLM_ENABLE_PROFILE_API",
        ),
    )


SETTINGS = load_settings()

# Backward-compatible aliases used throughout the current repository.
MODEL_ID = SETTINGS.model_id
DEVICE = SETTINGS.device
DTYPE = SETTINGS.dtype
