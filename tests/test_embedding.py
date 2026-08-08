import pytest
import torch

from barellm.models.embedding import (
    RotaryEmbedding,
    apply_rotary_pos_emb,
    rotate_half,
)


class TestRotaryEmbedding:
    def test_cos_sin_shape(self) -> None:
        rope = RotaryEmbedding(head_dim=8)
        x = torch.randn(2, 4, 3, 8)  # [B, H, S, D_h]
        cos, sin = rope(x, position_ids=None)
        assert cos.shape == (2, 1, 3, 8)
        assert sin.shape == (2, 1, 3, 8)

    def test_invalid_head_dim_raises(self) -> None:
        with pytest.raises(ValueError):
            RotaryEmbedding(head_dim=7)

    def test_position_zero_is_identity(self) -> None:
        rope = RotaryEmbedding(head_dim=8)
        x = torch.randn(1, 2, 4, 8)
        position_ids = torch.zeros(1, 4, dtype=torch.long)
        cos, sin = rope(x, position_ids)
        assert torch.allclose(cos, torch.ones_like(cos))
        assert torch.allclose(sin, torch.zeros_like(sin))

    def test_default_position_ids(self) -> None:
        rope = RotaryEmbedding(head_dim=8)
        x = torch.randn(2, 1, 5, 8)
        cos, _sin = rope(x, position_ids=None)
        # position 0 should have cos=1 (no rotation)
        assert torch.allclose(cos[:, :, 0, :], torch.ones(2, 1, 8))


class TestRotateHalf:
    def test_rotate_half(self) -> None:
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = rotate_half(x)
        # [-x2, x1] = [-3, -4, 1, 2]
        assert torch.allclose(result, torch.tensor([-3.0, -4.0, 1.0, 2.0]))


class TestApplyRotaryPosEmb:
    def test_preserves_shape(self) -> None:
        rope = RotaryEmbedding(head_dim=8)
        q = torch.randn(2, 4, 3, 8)
        k = torch.randn(2, 4, 3, 8)
        cos, sin = rope(q, position_ids=None)
        q_emb, k_emb = apply_rotary_pos_emb(q, k, cos, sin)
        assert q_emb.shape == q.shape
        assert k_emb.shape == k.shape

    def test_position_zero_is_identity(self) -> None:
        rope = RotaryEmbedding(head_dim=8)
        q = torch.randn(1, 2, 3, 8)
        k = torch.randn(1, 2, 3, 8)
        position_ids = torch.zeros(1, 3, dtype=torch.long)
        cos, sin = rope(q, position_ids)
        q_emb, k_emb = apply_rotary_pos_emb(q, k, cos, sin)
        assert torch.allclose(q_emb, q)
        assert torch.allclose(k_emb, k)

    def test_preserves_norm(self) -> None:
        rope = RotaryEmbedding(head_dim=8)
        q = torch.randn(1, 2, 4, 8)
        k = torch.randn(1, 2, 4, 8)
        cos, sin = rope(q, position_ids=None)
        q_emb, k_emb = apply_rotary_pos_emb(q, k, cos, sin)
        # RoPE is orthogonal: each dim-pair's norm is conserved
        assert torch.allclose(q.norm(dim=-1), q_emb.norm(dim=-1), atol=1e-5)
        assert torch.allclose(k.norm(dim=-1), k_emb.norm(dim=-1), atol=1e-5)

    def test_dot_product_depends_on_relative_position(self) -> None:
        # RoPE core property: <q_m, k_n> depends only on (m - n), not absolute
        rope = RotaryEmbedding(head_dim=8)
        q = torch.randn(1, 1, 1, 8)
        k = torch.randn(1, 1, 1, 8)

        # cos/sin for positions 0..15: [1, 1, 16, 8]
        dummy = torch.randn(1, 1, 16, 8)
        cos, sin = rope(dummy, position_ids=None)

        def score(q_pos: int, k_pos: int) -> torch.Tensor:
            qr = (
                q * cos[:, :, q_pos : q_pos + 1]
                + rotate_half(q) * sin[:, :, q_pos : q_pos + 1]
            )
            kr = (
                k * cos[:, :, k_pos : k_pos + 1]
                + rotate_half(k) * sin[:, :, k_pos : k_pos + 1]
            )
            return (qr * kr).sum(dim=-1)

        # Same relative distance (2 apart), different absolute positions
        assert torch.allclose(score(1, 3), score(5, 7), atol=1e-5)
        assert torch.allclose(score(0, 2), score(10, 12), atol=1e-5)
