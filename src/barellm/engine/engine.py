import logging
import time

import torch

from barellm.engine.kv_cache_manager import KVCacheManager
from barellm.engine.request import Request, RequestStatus
from barellm.engine.scheduler import Scheduler
from barellm.sampling.sampler import sample
from barellm.sampling.stops import FINISH_ABORT, FINISH_LENGTH

logger = logging.getLogger(__name__)


class Engine:
    def __init__(
        self,
        model: torch.nn.Module,
        scheduler: Scheduler,
        kv_cache_manager: KVCacheManager,
    ):
        self.model = model
        self.scheduler = scheduler
        self.kv_cache_manager = kv_cache_manager
        self.step_count = 0
        self.device = next(self.model.parameters()).device

    def _check_and_finish(self, req: Request):
        """Check stop conditions, fire callbacks."""
        on_token_streamed = False

        result = req.check_stop()

        if result is not None:
            req.finish_reason, req.stop_reason = result
            req.status = RequestStatus.FINISHED

        if req.on_token:
            token_id = int(req.token_ids[0, -1].item())
            should_continue = req.on_token(token_id, req.seq_len)
            on_token_streamed = True
            if should_continue is False:
                req.status = RequestStatus.FINISHED
                req.stop_reason = None
                req.finish_reason = FINISH_ABORT

        if req.status == RequestStatus.FINISHED:
            if req.on_token and not on_token_streamed:
                token_id = int(req.token_ids[0, -1].item())
                req.on_token(token_id, req.seq_len)

            if req.on_finish:
                req.on_finish(req.finish_reason or FINISH_LENGTH, req.stop_reason)

    def _prefill(self, req: Request):
        logger.debug("Prefill Request: %s", req.id)
        if req.max_new_tokens <= 0:
            req.finish_reason = FINISH_LENGTH
            req.status = RequestStatus.FINISHED
            if req.on_finish:
                req.on_finish(FINISH_LENGTH, None)
            return

        req.token_ids = req.token_ids.to(self.device)
        position_ids = torch.arange(
            req.seq_len, dtype=torch.long, device=self.device
        ).unsqueeze(0)  # [1, T]

        cache = self.kv_cache_manager.get_cache(req)

        logits = self.model(
            req.token_ids,
            position_ids=position_ids,
            kv_cache=cache,
        )  # [1, T, vocab]

        next_token = sample(
            logits[:, -1, :], req.temperature, req.top_k, req.top_p
        )  # [1, 1]
        req.append(next_token)

        self._check_and_finish(req)

    def _decode(self):
        logger.debug(
            "Decode: running=%d, waiting=%d",
            len(self.scheduler.running),
            len(self.scheduler.waiting),
        )

        active = self.scheduler.running
        if not active:
            return

        for req in active:
            self.kv_cache_manager.allocate_request(req)
            cache = self.kv_cache_manager.get_cache(req)

            input_ids = req.token_ids[:, -1:]
            position_ids = torch.tensor(
                [[req.seq_len - 1]], dtype=torch.long, device=self.device
            )

            logits = self.model(
                input_ids,
                position_ids=position_ids,
                kv_cache=cache,
            )  # [B, T, vocab]

            token = sample(
                logits[:, -1, :], req.temperature, req.top_k, req.top_p
            )  # [1, 1]
            req.append(token)
            self._check_and_finish(req)

    def _cleanup(self):
        logger.debug(
            "Clean Up: running=%d, waiting=%d",
            len(self.scheduler.running),
            len(self.scheduler.waiting),
        )
        newly_finished = self.scheduler.evict_finished()
        for req in newly_finished:
            if self.kv_cache_manager is not None:
                self.kv_cache_manager.free_request(req.id)

    def step(self) -> None:
        self.step_count += 1
        logger.debug(
            "Step %s: running=%d, waiting=%d - device=%s",
            self.step_count,
            len(self.scheduler.running),
            len(self.scheduler.waiting),
            self.device,
        )
        # 0. Check for new requests and prefill sequentially
        for req in self.scheduler.get_candidates():
            if not self.kv_cache_manager.allocate_request(req):
                break
            self.scheduler.pop_request(req)
            self._prefill(req)
            self.scheduler.start_request(req)
        # 1. Decode all active requests
        self._decode()
        # 2. Clean up finished requests
        self._cleanup()

    def run(
        self,
        max_steps: int | None = None,
        timeout: float | None = None,
    ) -> None:
        start_time = time.time()
        step = 0
        while self.scheduler.has_work():
            if max_steps and step >= max_steps:
                break
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError("Engine run exceeded timeout")
            with torch.inference_mode():
                self.step()
            step += 1
