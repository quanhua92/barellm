from dataclasses import dataclass
from enum import Enum

import torch


class RequestStatus(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass
class Request:
    id: str
    token_ids: torch.Tensor  # [1, T]
    status: RequestStatus = RequestStatus.WAITING

    @property
    def seq_len(self) -> int:
        return self.token_ids.shape[1]

    def append(self, token: torch.Tensor) -> None:
        self.token_ids = torch.cat((self.token_ids, token), dim=1)
