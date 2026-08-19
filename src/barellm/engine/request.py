from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import torch

from barellm.sampling.stops import check_stop


class RequestStatus(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


DecodeFunction = Callable[[list[int]], str]  # decode_fn(token_ids: list[int] ) -> str

TokenCallback = Callable[
    [int, int], bool | None
]  # on_token(token_id: int, count:int) -> bool | None) (returns False to abort)

FinishCallback = Callable[
    [str, int | str | None], None
]  # on_finish(finish_reason: str, stop_reason: int | str | None) -> None


@dataclass
class Request:
    id: str
    token_ids: torch.Tensor  # [1, T]
    status: RequestStatus = RequestStatus.WAITING
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    max_new_tokens: int = 512
    eos_ids: set[int] = field(default_factory=set)
    stop_strings: list[str] | None = None
    deadline: float | None = None
    decode_fn: DecodeFunction | None = None
    on_token: TokenCallback | None = None
    on_finish: FinishCallback | None = None
    finish_reason: str | None = None
    stop_reason: int | str | None = None
    generated_count: int = 0

    @property
    def seq_len(self) -> int:
        return self.token_ids.shape[1]

    def append(self, token: torch.Tensor) -> None:
        self.token_ids = torch.cat((self.token_ids, token), dim=1)
        self.generated_count += 1

    def check_stop(self) -> tuple[str, int | str | None] | None:
        token_id = int(self.token_ids[0, -1].item())
        text_so_far = ""
        if self.stop_strings and self.decode_fn:
            text_so_far = self.decode_fn(self.token_ids[0].tolist())

        return check_stop(
            token_id,
            self.eos_ids,
            self.generated_count,
            self.max_new_tokens,
            text_so_far,
            self.stop_strings,
            self.deadline,
        )
