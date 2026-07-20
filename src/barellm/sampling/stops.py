import time

FINISH_STOP = "stop"  # EOS token or stop string matched
FINISH_LENGTH = "length"  # max_tokens reached
FINISH_ABORT = "abort"  # cancelled, deadline exceeded


def check_stop(
    token_id: int,
    eos_ids: set[int],
    generated_count: int,
    max_new_tokens: int,
    text_so_far: str,
    stop_strings: list[str] | None = None,
    deadline: float | None = None,
) -> tuple[str, int | str | None] | None:
    """Check all stop conditions.

    Returns (finish_reason, stop_reason) if should stop, None if continue.
    finish_reason: "stop", "length", or "abort"
    stop_reason: specific token ID or stop string that triggered, or None
    """
    # 1. EOS token
    if token_id in eos_ids:
        return (FINISH_STOP, token_id)

    # 2. Max tokens - hard limit
    if generated_count >= max_new_tokens:
        return (FINISH_LENGTH, None)

    # 3. Stop string - user-defined phrase matched
    if stop_strings:
        for s in stop_strings:
            if s in text_so_far:
                return (FINISH_STOP, s)

    # 4. Deadline - safety net
    if deadline is not None and time.monotonic() > deadline:
        return (FINISH_ABORT, None)

    return None
