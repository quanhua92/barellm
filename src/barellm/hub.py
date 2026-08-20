import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"

_path_cache: dict[str, Path] = {}


def has_weights(model_dir: Path) -> bool:
    snapshots = model_dir / "snapshots"
    if not snapshots.exists():
        return False
    return any(snapshots.glob("*/**/*.safetensors")) or any(
        snapshots.glob("*/**.*.bin")
    )


def download_model(model_id: str, revision: str = "main") -> Path:
    """
    Download a model from Hugging Face Hub and return the local path to the model directory.

    Args:
        model_id (str): The model identifier on Hugging Face Hub (e.g., "Qwen/Qwen3-0.6B").
        revision (str): The specific revision of the model to download (default is "main").

    Returns:
    """
    cache_key = f"{model_id}:{revision}"
    if cache_key in _path_cache:
        return _path_cache[cache_key]
    model_path = snapshot_download(
        repo_id=model_id, revision=revision, cache_dir=HF_CACHE
    )
    _path_cache[cache_key] = Path(model_path)
    return Path(model_path)


def list_models() -> list[tuple[str, bool]]:
    """
    List all models in the local Hugging Face cache.
    """
    if not HF_CACHE.exists():
        return []

    models = []
    for model_dir in HF_CACHE.iterdir():
        name = model_dir.name
        if model_dir.is_dir() and name.startswith("models--"):
            name = name.replace("models--", "").replace("--", "/")
            ready = has_weights(model_dir)
            models.append((name, ready))
    return sorted(models)


def remove_model(model_id: str) -> bool:
    """
    Remove a model from the local Hugging Face cache.

    Args:
        model_id (str): The model identifier to remove (e.g., "Qwen/Qwen3-0.6B").

    Returns:
        bool: True if the model was successfully removed, False if the model was not found.
    """

    model_dir = HF_CACHE / f"models--{model_id.replace('/', '--')}"
    if model_dir.exists() and model_dir.is_dir():
        shutil.rmtree(model_dir)
        return True
    return False
