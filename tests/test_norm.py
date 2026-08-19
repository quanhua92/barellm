import torch

from barellm.models.norm import RMSNorm


def test_preserves_shape():
    norm = RMSNorm(dim=8)
    x = torch.randn(2, 4, 8)

    y = norm(x)

    assert y.shape == x.shape


def test_preserves_dtype():
    norm = RMSNorm(dim=8)
    x = torch.randn(2, 4, 8, dtype=torch.float32)

    y = norm(x)

    assert y.dtype == x.dtype


def test_output_is_finite():
    norm = RMSNorm(dim=8)
    x = torch.randn(2, 4, 8)

    y = norm(x)

    assert torch.isfinite(y).all()


def test_normalizes_rms():
    norm = RMSNorm(dim=8)
    x = torch.randn(2, 4, 8)

    y = norm(x)

    rms = torch.sqrt(y.pow(2).mean(dim=-1))

    assert torch.allclose(
        rms,
        torch.ones_like(rms),
        atol=1e-5,
    )


def test_weight_is_learnable():
    norm = RMSNorm(dim=8)

    assert isinstance(norm.weight, torch.nn.Parameter)
    assert norm.weight.requires_grad


def test_gradient_flows():
    norm = RMSNorm(dim=8)
    x = torch.randn(2, 4, 8, requires_grad=True)

    y = norm(x)
    loss = y.sum()
    loss.backward()

    assert x.grad is not None
    assert norm.weight.grad is not None
