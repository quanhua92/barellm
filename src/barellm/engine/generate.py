from dataclasses import dataclass
from uuid import uuid4

import torch

from barellm.engine.engine import Engine
from barellm.engine.request import (
    DecodeFunction,
    FinishCallback,
    Request,
    TokenCallback,
)
from barellm.utils import check


@dataclass(frozen=True)
class GenerationResult:
    token_ids: torch.Tensor
    finish_reason: str | None
    stop_reason: int | str | None
    generated_count: int

    @property
    def prompt_length(self) -> int:
        return self.token_ids.shape[1] - self.generated_count

    @property
    def generated_token_ids(self) -> torch.Tensor:
        if self.generated_count == 0:
            return self.token_ids[:, :0]

        return self.token_ids[:, -self.generated_count :]


def generate(
    engine: Engine,
    token_ids: torch.Tensor,
    *,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    eos_ids: set[int] | None = None,
    stop_strings: list[str] | None = None,
    decode_fn: DecodeFunction | None = None,
    on_token: TokenCallback | None = None,
    on_finish: FinishCallback | None = None,
    request_id: str | None = None,
    deadline: float | None = None,
) -> GenerationResult:
    check(
        token_ids.ndim == 2,
        "token_ids must have shape [1, T]",
    )
    check(
        token_ids.shape[0] == 1,
        "generate currently supports batch size 1",
    )
    check(
        token_ids.shape[1] > 0,
        "token_ids must contain at least one prompt token",
    )
    check(
        token_ids.dtype in (torch.int32, torch.int64),
        "token_ids must contain integer token IDs",
    )
    check(
        max_new_tokens >= 0,
        "max_new_tokens must be non-negative",
    )
    check(
        temperature >= 0.0,
        "temperature must be non-negative",
    )
    check(
        top_k >= 0,
        "top_k must be non-negative",
    )
    check(
        0.0 < top_p <= 1.0,
        "top_p must be in the range (0, 1]",
    )
    check(
        not stop_strings or decode_fn is not None,
        "decode_fn is required when stop_strings are provided",
    )
    check(
        request_id is None or bool(request_id),
        "request_id must not be empty",
    )
    check(
        not engine.scheduler.waiting and not engine.scheduler.running,
        "generate requires an idle engine",
    )

    request = Request(
        id=request_id or f"generate-{uuid4().hex}",
        token_ids=token_ids.clone(),
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        eos_ids=set() if eos_ids is None else set(eos_ids),
        stop_strings=None if stop_strings is None else list(stop_strings),
        decode_fn=decode_fn,
        on_token=on_token,
        on_finish=on_finish,
        deadline=deadline,
    )
    engine.scheduler.add_request(request)
    engine.run()

    return GenerationResult(
        token_ids=request.token_ids,
        finish_reason=request.finish_reason,
        stop_reason=request.stop_reason,
        generated_count=request.generated_count,
    )
