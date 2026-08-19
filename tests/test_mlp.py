import pytest
import torch

from barellm.models.mlp import SwiGLU


def test_preserves_shape():
    mlp = SwiGLU(hidden_size=16, intermediate_size=32)
    x = torch.randn(2, 4, 16)

    y = mlp(x)

    assert y.shape == x.shape


def test_output_is_finite():
    mlp = SwiGLU(hidden_size=16, intermediate_size=32)
    x = torch.randn(2, 4, 16)

    y = mlp(x)

    assert torch.isfinite(y).all()


def test_projections_have_no_bias():
    mlp = SwiGLU(hidden_size=16, intermediate_size=32)

    assert mlp.gate_proj.bias is None
    assert mlp.up_proj.bias is None
    assert mlp.down_proj.bias is None


def test_rejects_invalid_dimensions():
    with pytest.raises(ValueError):
        SwiGLU(hidden_size=0, intermediate_size=32)

    with pytest.raises(ValueError):
        SwiGLU(hidden_size=16, intermediate_size=0)


def test_gradient_flows():
    mlp = SwiGLU(hidden_size=16, intermediate_size=32)
    x = torch.randn(2, 4, 16, requires_grad=True)

    y = mlp(x)
    y.sum().backward()

    assert x.grad is not None
    assert mlp.gate_proj.weight.grad is not None
    assert mlp.up_proj.weight.grad is not None
    assert mlp.down_proj.weight.grad is not None
