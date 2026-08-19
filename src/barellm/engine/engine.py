import logging
import time

import torch

from barellm.engine.kv_cache_manager import KVCacheManager
from barellm.engine.request import Request, RequestStatus
from barellm.engine.scheduler import Scheduler
from barellm.sampling.stops import FINISH_ABORT, FINISH_LENGTH

logger = logging.getLogger(__name__)


class Engine:
    def __init__(self, scheduler: Scheduler, kv_cache_manager: KVCacheManager):
        self.scheduler = scheduler
        self.kv_cache_manager = kv_cache_manager
        self.step_count = 0

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
        # 0. logits = self.model(req.token_ids, kv_cache_manager)
        # 1. next_token = sample(logits[:, -1, :], ...)
        # 2. req.append(next_token)
        # 3. _check_and_finish

    def _decode(self):
        logger.debug("Decode")

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

    def step(self):
        self.step_count += 1
        logger.debug(
            "Step %s: running=%d, waiting=%d",
            self.step_count,
            len(self.scheduler.running),
            len(self.scheduler.waiting),
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

    def run(self, max_steps: int | None = None, timeout: float | None = None):
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
