import pytest
import torch

from barellm.models.embedding import TiedLMHead, TokenEmbedding


def test_embedding_shape():
    embedding = TokenEmbedding(vocab_size=100, hidden_size=16)
    token_ids = torch.tensor([[1, 2, 3]])

    output = embedding(token_ids)

    assert output.shape == (1, 3, 16)


def test_embedding_supports_batches():
    embedding = TokenEmbedding(vocab_size=100, hidden_size=16)
    token_ids = torch.tensor([[1, 2], [3, 4]])

    output = embedding(token_ids)

    assert output.shape == (2, 2, 16)
    assert torch.equal(output[0, 0], embedding.weight[1])


def test_embedding_rejects_invalid_dimensions():
    with pytest.raises(ValueError):
        TokenEmbedding(vocab_size=0, hidden_size=16)

    with pytest.raises(ValueError):
        TokenEmbedding(vocab_size=100, hidden_size=0)


def test_lm_head_is_tied():
    embedding = TokenEmbedding(vocab_size=100, hidden_size=16)
    head = TiedLMHead(embedding)

    assert head.weight is embedding.weight


def test_lm_head_shape():
    embedding = TokenEmbedding(vocab_size=100, hidden_size=16)
    head = TiedLMHead(embedding)
    hidden = torch.randn(2, 4, 16)

    logits = head(hidden)

    assert logits.shape == (2, 4, 100)


def test_lm_head_uses_embedding_weights():
    embedding = TokenEmbedding(vocab_size=3, hidden_size=2)
    head = TiedLMHead(embedding)
    hidden = torch.tensor([[[1.0, 2.0]]])

    with torch.no_grad():
        embedding.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [2.0, 3.0],
                ]
            )
        )

    logits = head(hidden)

    assert torch.equal(logits, torch.tensor([[[1.0, 2.0, 8.0]]]))
