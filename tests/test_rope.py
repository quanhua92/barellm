import pytest
import torch

from barellm.models.rope import RotaryEmbedding, apply_rope, rotate_half


def test_rejects_odd_head_dim():
    with pytest.raises(ValueError):
        RotaryEmbedding(head_dim=3)


def test_cos_sin_shapes():
    rope = RotaryEmbedding(head_dim=8)
    x = torch.randn(2, 4, 6, 8)
    position_ids = torch.arange(6).unsqueeze(0).expand(2, -1)

    cos, sin = rope(x, position_ids)

    assert cos.shape == (2, 1, 6, 8)
    assert sin.shape == (2, 1, 6, 8)


def test_position_zero_is_identity():
    rope = RotaryEmbedding(head_dim=4)
    x = torch.randn(1, 1, 1, 4)
    position_ids = torch.zeros(1, 1, dtype=torch.long)

    cos, sin = rope(x, position_ids)
    output = apply_rope(x, cos, sin)

    assert torch.allclose(output, x)


def test_half_split_pairing():
    x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])

    rotated = rotate_half(x)

    # Pairing is (x0, x2) and (x1, x3), not (x0, x1) and (x2, x3).
    expected = torch.tensor([[[[-3.0, -4.0, 1.0, 2.0]]]])

    assert torch.equal(rotated, expected)


def test_known_half_split_rotation():
    x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])

    # Rotate both pairs by 90 degrees.
    cos = torch.zeros(1, 1, 1, 4)
    sin = torch.ones(1, 1, 1, 4)

    output = apply_rope(x, cos, sin)

    expected = torch.tensor([[[[-3.0, -4.0, 1.0, 2.0]]]])

    assert torch.allclose(output, expected)


def test_cos_sin_are_repeated_for_half_split_pairs():
    rope = RotaryEmbedding(head_dim=4, theta=100.0)
    x = torch.zeros(1, 1, 2, 4)
    position_ids = torch.tensor([[0, 1]])

    cos, sin = rope(x, position_ids)

    assert torch.equal(cos[:, :, :, :2], cos[:, :, :, 2:])
    assert torch.equal(sin[:, :, :, :2], sin[:, :, :, 2:])


def test_positions_change_rotation():
    rope = RotaryEmbedding(head_dim=4)
    x = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]]]])

    position_zero = torch.zeros(1, 1, dtype=torch.long)
    position_one = torch.ones(1, 1, dtype=torch.long)

    cos_zero, sin_zero = rope(x, position_zero)
    cos_one, sin_one = rope(x, position_one)

    output_zero = apply_rope(x, cos_zero, sin_zero)
    output_one = apply_rope(x, cos_one, sin_one)

    assert torch.allclose(output_zero, x)
    assert not torch.allclose(output_zero, output_one)


def test_default_positions_match_explicit_positions():
    rope = RotaryEmbedding(head_dim=8)
    x = torch.randn(2, 3, 4, 8)
    position_ids = torch.arange(4).unsqueeze(0).expand(2, -1)

    default_cos, default_sin = rope(x)
    explicit_cos, explicit_sin = rope(x, position_ids)

    assert torch.allclose(default_cos, explicit_cos)
    assert torch.allclose(default_sin, explicit_sin)
