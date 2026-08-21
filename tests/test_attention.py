from typing import Any

import pytest
import torch

import barellm.models.attention as attention_module
from barellm.models.attention import CausalSelfAttention, GroupedQueryAttention


def make_attention(**kwargs):
    defaults: dict[str, Any] = {
        "hidden_size": 16,
        "head_dim": 4,
        "num_heads": 4,
    }
    defaults.update(kwargs)
    return CausalSelfAttention(**defaults)


def make_gqa(**kwargs):
    defaults: dict[str, Any] = {
        "hidden_size": 16,
        "head_dim": 4,
        "num_heads": 4,
        "num_kv_heads": 2,
    }
    defaults.update(kwargs)
    return GroupedQueryAttention(**defaults)


def test_preserves_shape():
    attention = make_attention()
    x = torch.randn(2, 8, 16)

    output = attention(x)

    assert output.shape == x.shape


def test_output_is_finite():
    attention = make_attention()
    x = torch.randn(2, 8, 16)

    output = attention(x)

    assert torch.isfinite(output).all()


def test_causal_masking():
    torch.manual_seed(42)
    attention = make_attention()
    x = torch.randn(1, 8, 16)

    output_before = attention(x)

    x_after = x.clone()
    x_after[:, 4:, :] += 10.0
    output_after = attention(x_after)

    # Earlier positions cannot attend to later positions.
    assert torch.allclose(
        output_before[:, :4, :],
        output_after[:, :4, :],
        atol=1e-6,
    )


def test_explicit_positions_preserve_shape():
    attention = make_attention()
    x = torch.randn(2, 5, 16)
    position_ids = torch.arange(5).unsqueeze(0).expand(2, -1)

    output = attention(x, position_ids=position_ids)

    assert output.shape == x.shape


def test_qk_norm_preserves_shape():
    attention = make_attention(use_qk_norm=True)
    x = torch.randn(2, 8, 16)

    output = attention(x)

    assert output.shape == x.shape


def test_qk_norm_changes_output():
    torch.manual_seed(42)
    without_norm = make_attention(use_qk_norm=False)
    torch.manual_seed(42)
    with_norm = make_attention(use_qk_norm=True)
    x = torch.randn(1, 8, 16)

    output_without = without_norm(x)
    output_with = with_norm(x)

    assert not torch.allclose(output_without, output_with)


def test_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        make_attention(hidden_size=0)

    with pytest.raises(ValueError):
        make_attention(num_heads=0)

    with pytest.raises(ValueError):
        make_attention(head_dim=3)


def test_gradient_flows():
    attention = make_attention()
    x = torch.randn(2, 4, 16, requires_grad=True)

    output = attention(x)
    output.sum().backward()

    assert x.grad is not None
    assert attention.q_proj.weight.grad is not None
    assert attention.k_proj.weight.grad is not None
    assert attention.v_proj.weight.grad is not None
    assert attention.o_proj.weight.grad is not None


def test_gqa_preserves_shape():
    attention = make_gqa()
    x = torch.randn(2, 8, 16)

    output = attention(x)

    assert output.shape == x.shape


def test_mqa_preserves_shape():
    attention = make_gqa(num_kv_heads=1)
    x = torch.randn(2, 8, 16)

    output = attention(x)

    assert output.shape == x.shape


@pytest.mark.parametrize(
    "num_kv_heads, group_size",
    [(4, 1), (2, 2), (1, 4)],
    ids=["mha", "gqa", "mqa"],
)
def test_kv_projection_shapes_match_head_layout(
    num_kv_heads: int,
    group_size: int,
) -> None:
    attention = make_gqa(num_kv_heads=num_kv_heads)

    assert attention.q_proj.out_features == 4 * 4
    assert attention.k_proj.out_features == num_kv_heads * 4
    assert attention.v_proj.out_features == num_kv_heads * 4
    assert attention.o_proj.in_features == 4 * 4
    assert attention.group_size == group_size


def test_gqa_repeats_each_kv_head_for_its_query_group(monkeypatch):
    attention = make_gqa(num_kv_heads=2)
    x = torch.randn(1, 3, 16)
    captured: dict[str, torch.Tensor] = {}

    def identity_rope(x, cos, sin):
        return x

    def capture_attention(q, k, v, **_kwargs):
        captured["q"] = q
        captured["k"] = k
        captured["v"] = v
        return torch.zeros_like(q)

    monkeypatch.setattr(attention_module, "apply_rope", identity_rope)
    monkeypatch.setattr(
        attention_module.F,
        "scaled_dot_product_attention",
        capture_attention,
    )

    attention(x)

    raw_k = attention.k_proj(x).view(1, 3, 2, 4).transpose(1, 2)
    raw_v = attention.v_proj(x).view(1, 3, 2, 4).transpose(1, 2)

    assert captured["q"].shape == (1, 4, 3, 4)
    torch.testing.assert_close(captured["k"], raw_k.repeat_interleave(2, dim=1))
    torch.testing.assert_close(captured["v"], raw_v.repeat_interleave(2, dim=1))


def test_gqa_output_is_finite():
    attention = make_gqa()
    x = torch.randn(2, 8, 16)

    output = attention(x)

    assert torch.isfinite(output).all()


def test_gqa_matches_mha_when_kv_heads_equal_query_heads():
    torch.manual_seed(42)
    mha = make_attention()
    gqa = make_gqa(num_kv_heads=4)
    gqa.load_state_dict(mha.state_dict())

    x = torch.randn(2, 8, 16)

    output_mha = mha(x)
    output_gqa = gqa(x)

    assert torch.allclose(output_mha, output_gqa)


def test_gqa_causal_masking():
    torch.manual_seed(42)
    attention = make_gqa()
    x = torch.randn(1, 8, 16)

    output_before = attention(x)

    x_after = x.clone()
    x_after[:, 4:, :] += 10.0
    output_after = attention(x_after)

    assert torch.allclose(
        output_before[:, :4, :],
        output_after[:, :4, :],
        atol=1e-6,
    )


def test_gqa_qk_norm_preserves_shape():
    attention = make_gqa(use_qk_norm=True)
    x = torch.randn(2, 8, 16)

    output = attention(x)

    assert output.shape == x.shape


@pytest.mark.parametrize("num_heads, num_kv_heads", [(4, 3), (4, 5), (4, 0)])
def test_gqa_rejects_invalid_head_grouping(num_heads: int, num_kv_heads: int):
    with pytest.raises(ValueError):
        make_gqa(num_heads=num_heads, num_kv_heads=num_kv_heads)
