import torch

from barellm.models.attention import GroupedQueryAttention
from barellm.models.qwen3 import Qwen3Config, Qwen3ForCausalLM


def make_small_model() -> Qwen3ForCausalLM:
    return Qwen3ForCausalLM(
        vocab_size=100,
        hidden_size=16,
        intermediate_size=32,
        head_dim=4,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        use_qk_norm=True,
    )


def make_config() -> Qwen3Config:
    return Qwen3Config(
        model_type="qwen3",
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        intermediate_size=32,
        vocab_size=100,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        max_position_embeddings=128,
        rope_scaling=None,
        tie_word_embeddings=True,
        bos_token_id=0,
        eos_token_id=1,
        hidden_act="silu",
        attention_bias=False,
        attention_dropout=0.0,
        use_sliding_window=False,
    )


def test_preserves_logits_shape():
    model = make_small_model()
    token_ids = torch.randint(0, 100, (2, 8))

    logits = model(token_ids)

    assert logits.shape == (2, 8, 100)


def test_output_is_finite():
    model = make_small_model()
    token_ids = torch.randint(0, 100, (2, 8))

    logits = model(token_ids)

    assert torch.isfinite(logits).all()


def test_uses_grouped_query_attention():
    model = make_small_model()

    for layer in model.layers:
        assert isinstance(layer.self_attn, GroupedQueryAttention)
        assert layer.self_attn.num_kv_heads == 2


def test_has_expected_number_of_layers():
    model = make_small_model()

    assert len(model.layers) == 2


def test_lm_head_uses_tied_embedding_weights():
    model = make_small_model()

    assert model.lm_head.weight is model.embed_tokens.weight


def test_supports_explicit_positions():
    model = make_small_model()
    token_ids = torch.randint(0, 100, (2, 8))
    position_ids = torch.arange(8).unsqueeze(0).expand(2, -1)

    logits = model(token_ids, position_ids=position_ids)

    assert logits.shape == (2, 8, 100)


def test_from_config_wires_model_dimensions():
    config = make_config()
    model = Qwen3ForCausalLM.from_config(config)

    assert len(model.layers) == config.num_hidden_layers
    assert model.layers[0].self_attn.num_kv_heads == config.num_key_value_heads
    assert model.layers[0].self_attn.use_qk_norm is True
    assert model.layers[0].self_attn.rotary_emb.theta == config.rope_theta


def test_from_config_forward():
    model = Qwen3ForCausalLM.from_config(make_config())
    token_ids = torch.randint(0, 100, (2, 8))

    logits = model(token_ids)

    assert logits.shape == (2, 8, 100)
    assert torch.isfinite(logits).all()
