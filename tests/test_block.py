from typing import Any

import pytest
import torch

from barellm.models.block import TransformerBlock


def make_block(**kwargs):
    defaults: dict[str, Any] = {
        "hidden_size": 16,
        "head_dim": 4,
        "num_heads": 4,
        "intermediate_size": 32,
    }
    defaults.update(kwargs)
    return TransformerBlock(**defaults)


def test_preserves_shape():
    block = make_block()
    x = torch.randn(2, 8, 16)

    output = block(x)

    assert output.shape == x.shape


def test_output_is_finite():
    block = make_block()
    x = torch.randn(2, 8, 16)

    output = block(x)

    assert torch.isfinite(output).all()


def test_causal_behavior():
    torch.manual_seed(42)
    block = make_block()
    x = torch.randn(1, 8, 16)

    output_before = block(x)

    x_after = x.clone()
    x_after[:, 4:, :] += 10.0
    output_after = block(x_after)

    assert torch.allclose(
        output_before[:, :4, :],
        output_after[:, :4, :],
        atol=1e-6,
    )


def test_qk_norm_preserves_shape():
    block = make_block(use_qk_norm=True)
    x = torch.randn(2, 8, 16)

    output = block(x)

    assert output.shape == x.shape


def test_residual_connections_change_input():
    block = make_block()
    x = torch.randn(2, 8, 16)

    output = block(x)

    assert not torch.allclose(output, x)


def test_gradient_flows():
    block = make_block()
    x = torch.randn(2, 4, 16, requires_grad=True)

    output = block(x)
    output.sum().backward()

    assert x.grad is not None
    assert block.input_layernorm.weight.grad is not None
    assert block.self_attn.q_proj.weight.grad is not None
    assert block.mlp.down_proj.weight.grad is not None


def test_rejects_invalid_dimensions():
    with pytest.raises(ValueError):
        make_block(hidden_size=0)

    with pytest.raises(ValueError):
        make_block(num_heads=0)

    with pytest.raises(ValueError):
        make_block(head_dim=3)

    with pytest.raises(ValueError):
        make_block(intermediate_size=0)
