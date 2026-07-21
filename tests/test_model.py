from __future__ import annotations

from collections import Counter

import pytest
import torch

from barellm.config import DTYPE
from barellm.model.weights import load_weights


@pytest.fixture(scope="module")
def weights() -> dict[str, torch.Tensor]:
    try:
        return load_weights()
    except Exception as e:
        pytest.skip(f"Qwen3-0.6B not available locally: {e}")


class TestLoadWeights:
    def test_returns_311_tensors(self, weights: dict[str, torch.Tensor]) -> None:
        assert len(weights) == 311

    def test_root_has_three_keys(self, weights: dict[str, torch.Tensor]) -> None:
        root_keys = [k for k in weights if ".layers." not in k]
        assert sorted(root_keys) == [
            "lm_head.weight",
            "model.embed_tokens.weight",
            "model.norm.weight",
        ]

    def test_layer_0_has_11_tensors(self, weights: dict[str, torch.Tensor]) -> None:
        layer0_keys = [k for k in weights if ".layers.0." in k]
        assert len(layer0_keys) == 11

    def test_all_28_layers_have_11_tensors(
        self, weights: dict[str, torch.Tensor]
    ) -> None:
        per_layer: Counter[str] = Counter()
        for k in weights:
            if ".layers." in k:
                layer_id = k.split(".layers.")[1].split(".")[0]
                per_layer[layer_id] += 1
        assert len(per_layer) == 28
        assert set(per_layer.values()) == {11}

    def test_qk_norm_present(self, weights: dict[str, torch.Tensor]) -> None:
        assert any(".q_norm." in k for k in weights)
        assert any(".k_norm." in k for k in weights)

    def test_all_tensors_correct_dtype(self, weights: dict[str, torch.Tensor]) -> None:
        bad = {k: t.dtype for k, t in weights.items() if t.dtype != DTYPE}
        assert not bad, f"Wrong dtype tensors: {bad}"

    def test_no_inf_or_nan(self, weights: dict[str, torch.Tensor]) -> None:
        for name, t in weights.items():
            assert not torch.isinf(t).any(), f"{name} has inf"
            assert not torch.isnan(t).any(), f"{name} has nan"

    def test_tied_embeddings(self, weights: dict[str, torch.Tensor]) -> None:
        assert torch.equal(
            weights["lm_head.weight"],
            weights["model.embed_tokens.weight"],
        )

    def test_expected_shapes(self, weights: dict[str, torch.Tensor]) -> None:
        assert weights["model.embed_tokens.weight"].shape == (151936, 1024)
        assert weights["model.norm.weight"].shape == (1024,)
        assert weights["model.layers.0.input_layernorm.weight"].shape == (1024,)
        assert weights["model.layers.0.post_attention_layernorm.weight"].shape == (
            1024,
        )
        assert weights["model.layers.0.self_attn.q_proj.weight"].shape == (
            2048,
            1024,
        )
        assert weights["model.layers.0.self_attn.k_proj.weight"].shape == (
            1024,
            1024,
        )
        assert weights["model.layers.0.self_attn.v_proj.weight"].shape == (
            1024,
            1024,
        )
        assert weights["model.layers.0.self_attn.o_proj.weight"].shape == (
            1024,
            2048,
        )
        assert weights["model.layers.0.self_attn.q_norm.weight"].shape == (128,)
        assert weights["model.layers.0.self_attn.k_norm.weight"].shape == (128,)
        assert weights["model.layers.0.mlp.gate_proj.weight"].shape == (3072, 1024)
        assert weights["model.layers.0.mlp.up_proj.weight"].shape == (3072, 1024)
        assert weights["model.layers.0.mlp.down_proj.weight"].shape == (1024, 3072)
