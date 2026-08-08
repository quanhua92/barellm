import pytest
import torch

from barellm.models.attention import CausalSelfAttention, GroupedQueryAttention


class TestCausalSelfAttention:
    def test_attention_preserves_shape(self) -> None:
        hidden_size = 16
        num_heads = 8
        attn = CausalSelfAttention(hidden_size, num_heads)
        B, S = 1, 16
        x = torch.randn(B, S, hidden_size)
        output = attn(x)
        assert output.shape == (B, S, hidden_size)

    def test_causal_masking(self) -> None:
        attn = CausalSelfAttention(16, 8)
        x = torch.randn(1, 8, 16)
        out1 = attn(x)
        x[:, 4:, :] += 10
        out2 = attn(x)
        # positions 0-3 can't attend to 4+ (causal), so they stay unchanged
        assert torch.allclose(out1[:, :4, :], out2[:, :4, :], atol=1e-6)

    def test_invalid_num_heads_raises(self) -> None:
        with pytest.raises(ValueError):
            CausalSelfAttention(16, 6)


class TestGroupedQueryAttention:
    def test_attention_preserves_shape(self) -> None:
        attn = GroupedQueryAttention(16, 8, num_kv_heads=2)
        B, S = 1, 16
        x = torch.randn(B, S, 16)
        output = attn(x)
        assert output.shape == (B, S, 16)

    def test_gqa_matches_mha_when_kv_heads_equal(self) -> None:
        D, H = 16, 8
        mha = CausalSelfAttention(D, H)
        gqa = GroupedQueryAttention(D, H, num_kv_heads=H)
        gqa.load_state_dict(mha.state_dict())
        x = torch.randn(1, 16, D)
        assert torch.allclose(mha(x), gqa(x))

    def test_causal_masking(self) -> None:
        attn = GroupedQueryAttention(16, 8, num_kv_heads=2)
        x = torch.randn(1, 8, 16)
        out1 = attn(x)
        x[:, 4:, :] += 10
        out2 = attn(x)
        # positions 0-3 can't attend to 4+ (causal), so they stay unchanged
        assert torch.allclose(out1[:, :4, :], out2[:, :4, :], atol=1e-6)

    def test_invalid_config_raises(self) -> None:
        with pytest.raises(ValueError):
            GroupedQueryAttention(16, 6, num_kv_heads=4)
