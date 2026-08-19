import torch
from torch import nn

from barellm.utils import check


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, std: float = 0.02):
        super().__init__()
        check(hidden_size > 0, "hidden_size must be positive")
        check(vocab_size > 0, "vocab_size must be positive")

        self.weight = nn.Parameter(torch.empty(vocab_size, hidden_size))
        nn.init.normal_(self.weight, mean=0, std=std)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: [B, T]
        # output: [B, T, D]
        return self.weight[token_ids]


class TiedLMHead(nn.Module):
    def __init__(self, token_embedding: TokenEmbedding):
        super().__init__()
        self.weight = token_embedding.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        # output: [B, T, vocab_size]
        return torch.matmul(x, self.weight.T)
