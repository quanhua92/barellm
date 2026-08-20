import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from barellm.models.weights import (
    default_map_key,
    load_into_model,
    load_weights,
)


def test_default_map_key_strips_model_prefix():
    assert default_map_key("model.layers.0.self_attn.q_proj.weight") == (
        "layers.0.self_attn.q_proj.weight"
    )
    assert default_map_key("embed_tokens.weight") == "embed_tokens.weight"


def test_load_weights_reads_shards(monkeypatch, tmp_path):
    save_file(
        {"model.first": torch.ones(2, 2)},
        str(tmp_path / "00001.safetensors"),
    )
    save_file(
        {"model.second": torch.zeros(2, 2)},
        str(tmp_path / "00002.safetensors"),
    )

    monkeypatch.setattr(
        "barellm.models.weights.download_model",
        lambda model_id: tmp_path,
    )

    weights = load_weights(
        model_id="test/model",
        device="cpu",
        dtype=torch.float64,
    )

    assert set(weights) == {"model.first", "model.second"}
    assert weights["model.first"].dtype == torch.float64
    assert torch.equal(weights["model.second"], torch.zeros(2, 2))


def test_load_weights_rejects_missing_safetensors(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "barellm.models.weights.download_model",
        lambda model_id: tmp_path,
    )

    with pytest.raises(FileNotFoundError):
        load_weights(
            model_id="test/model",
            device="cpu",
            dtype=torch.float32,
        )


def test_load_weights_rejects_duplicate_keys(monkeypatch, tmp_path):
    tensor = torch.ones(2, 2)
    save_file({"duplicate": tensor}, str(tmp_path / "00001.safetensors"))
    save_file({"duplicate": tensor}, str(tmp_path / "00002.safetensors"))

    monkeypatch.setattr(
        "barellm.models.weights.download_model",
        lambda model_id: tmp_path,
    )

    with pytest.raises(ValueError, match="Duplicate tensor key"):
        load_weights(
            model_id="test/model",
            device="cpu",
            dtype=torch.float32,
        )


def test_load_into_model_maps_and_loads_weights():
    model = nn.Linear(2, 2, bias=False)
    expected = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    load_into_model(
        model,
        {"model.weight": expected},
    )

    assert torch.equal(model.weight, expected)


def test_load_into_model_converts_dtype():
    model = nn.Linear(2, 2, bias=False)
    weights = {"weight": torch.ones(2, 2)}

    load_into_model(
        model,
        weights,
        dtype=torch.float64,
    )

    assert model.weight.dtype == torch.float64


def test_load_into_model_is_strict():
    model = nn.Linear(2, 2, bias=False)

    with pytest.raises(RuntimeError):
        load_into_model(
            model,
            {"model.missing": torch.ones(2, 2)},
        )
