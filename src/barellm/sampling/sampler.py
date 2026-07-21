import torch
import torch.nn.functional as F


def greedy(logits: torch.Tensor) -> torch.Tensor:
    """
    Greedy sampling: Select the token with the highest probability at each step.

    Args:
        logits (torch.Tensor): The logits output from the model of shape (batch_size, vocab_size).

    Returns:
        torch.Tensor: The indices of the selected tokens of shape (batch_size,).
    """
    return logits.argmax(dim=-1, keepdim=True)


def sample_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Sample from the logits using temperature scaling.

    Args:
        logits (torch.Tensor): The logits output from the model of shape (batch_size, vocab_size).
        temperature (float): The temperature value for scaling.

    Returns:
        torch.Tensor: The indices of the sampled tokens of shape (batch_size, 1).
    """
    if temperature <= 0:
        raise ValueError("Temperature must be a positive value.")
    scaled_logits = logits / max(temperature, 1e-8)  # Avoid division by zero
    probs = F.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def sample_top_k(
    logits: torch.Tensor, k: int, temperature: float = 0.7
) -> torch.Tensor:
    """
    Sample from the logits using top-k sampling with temperature scaling.

    Args:
        logits (torch.Tensor): The logits output from the model of shape (batch_size, vocab_size).
        k (int): The number of top tokens to consider for sampling.
        temperature (float): The temperature value for scaling.

    Returns:
        torch.Tensor: The indices of the sampled tokens of shape (batch_size, 1).
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    # Get the top-k logits and their indices
    top_k_logits, top_k_indices = torch.topk(logits, k=k, dim=-1)

    # Scale the top-k logits by temperature
    scaled_top_k_logits = top_k_logits / max(
        temperature, 1e-8
    )  # Avoid division by zero

    # Convert to probabilities
    probs = F.softmax(scaled_top_k_logits, dim=-1)

    # Sample from the top-k probabilities
    sampled_indices = torch.multinomial(probs, num_samples=1)

    # Map back to original indices
    return top_k_indices.gather(-1, sampled_indices)


def sample_top_p(
    logits: torch.Tensor, p: float, temperature: float = 0.7
) -> torch.Tensor:
    """
    Sample from the logits using top-p (nucleus) sampling with temperature scaling.

    Args:
        logits (torch.Tensor): The logits output from the model of shape (batch_size, vocab_size).
        p (float): The cumulative probability threshold for top-p sampling.
        temperature (float): The temperature value for scaling.

    Returns:
        torch.Tensor: The indices of the sampled tokens of shape (batch_size, 1).
    """
    if not (0 < p <= 1):
        raise ValueError("p must be in the range (0, 1].")

    # Scale logits by temperature
    scaled_logits = logits / max(temperature, 1e-8)  # Avoid division by zero

    # Convert to probabilities
    probs = F.softmax(scaled_logits, dim=-1)

    # Sort probabilities and their indices
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Create a mask for tokens to keep based on cumulative probability threshold p
    mask = cumulative_probs <= p

    # Ensure at least one token is kept
    mask[..., 0] = True

    # Mask out tokens that exceed the cumulative probability threshold
    filtered_probs = sorted_probs * mask.float()

    # Normalize the filtered probabilities
    filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)

    # Sample from the filtered probabilities
    sampled_indices = torch.multinomial(filtered_probs, num_samples=1)

    # Map back to original indices
    return sorted_indices.gather(-1, sampled_indices)


def sample(
    logits: torch.Tensor,
    temperature: float = 0.7,
    top_k: int = 0,
    top_p: float = 1.0,
):
    """greedy -> temperature -> top_k -> top_p -> multinomial"""
    if temperature <= 0.01:
        return greedy(logits)

    # 1. Temperature
    logits = logits / max(temperature, 1e-8)

    # 2. Top-k: mask everything below the k-th highest logit
    if top_k > 0:
        k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, k)
        logits = torch.where(
            logits < values[-1:], torch.full_like(logits, float("-inf")), logits
        )

    # 3. Top-p: mask tokens beyond cumulative probs threshold
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Keep tokens <= top_p, always keep at least one
        keep = cumulative_probs <= top_p
        keep[..., 0] = True

        # Map sorted keep-mask back to original vocab ordering, then invert
        remove = ~keep.scatter(-1, sorted_idx, keep)
        logits = logits.masked_fill(remove, float("-inf"))

    # 4. Softmax and sample
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
