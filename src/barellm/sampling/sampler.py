import torch
import torch.nn.functional as F


def sample(
    logits: torch.Tensor,  # [B, vocab]
    temperature: float = 1.0,
    top_k: int = 0,  # 0 = disabled
    top_p: float = 1.0,  # 1.0 = disabled
) -> torch.Tensor:
    """Sample next token from logits.

    Pipeline: greedy -> temperature -> top_k -> top_p -> multinomial
    """

    # 1. Greedy
    if temperature <= 0.01:
        return torch.argmax(logits, dim=-1, keepdim=True)

    # 2. Temperature
    logits = logits / max(temperature, 1e-8)  # [B, vocab]

    # 3. Top-k
    if top_k > 0:
        k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, k, dim=-1)  # [B, k]
        k_th_value = values[..., -1:]  # [B, 1]
        logits = logits.masked_fill(logits < k_th_value, float("-inf"))  # [B, vocab]

    # 4 Top-p
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)  # [B, vocab]
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1
        )  # [B, vocab]
        remove = cumulative_probs > top_p  # [B, vocab]

        # Shift right - always keep the first token
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False

        remove = remove.scatter(-1, sorted_idx, remove)  # [B, vocab]
        logits = logits.masked_fill(remove, float("-inf"))  # [B, vocab]

    # 5. Softmax + multinomial
    probs = F.softmax(logits, dim=-1)  # [B, vocab]
    return torch.multinomial(probs, num_samples=1)  # [B, 1]
