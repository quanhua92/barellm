import logging
import time

import torch

from barellm.engine.batched_kv_cache import BatchKVCache
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
        kv_cache_manager: KVCacheManager | None,
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

    def _prefill(self, req: Request, use_cache: bool) -> bool:
        logger.debug("Prefill Request: %s", req.id)
        if req.max_new_tokens <= 0:
            req.finish_reason = FINISH_LENGTH
            req.status = RequestStatus.FINISHED
            if req.on_finish:
                req.on_finish(FINISH_LENGTH, None)
            return True

        req.token_ids = req.token_ids.to(self.device)
        position_ids = torch.arange(
            req.seq_len, dtype=torch.long, device=self.device
        ).unsqueeze(0)  # [1, T]

        cache = None
        if use_cache:
            if self.kv_cache_manager is None:
                raise ValueError("cached execution requires a KV cache manager")
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
        return True

    def _decode(self, use_cache: bool = True) -> bool:
        logger.debug(
            "Decode: running=%d, waiting=%d",
            len(self.scheduler.running),
            len(self.scheduler.waiting),
        )

        active = self.scheduler.running
        if not active:
            return False

        if not use_cache:
            for req in active:
                position_ids = torch.arange(
                    req.seq_len,
                    dtype=torch.long,
                    device=self.device,
                ).unsqueeze(0)
                logits = self.model(
                    req.token_ids,
                    position_ids=position_ids,
                    kv_cache=None,
                )  # [1, T, vocab]
                token = sample(
                    logits[:, -1, :],
                    req.temperature,
                    req.top_k,
                    req.top_p,
                )  # [1, 1]
                req.append(token)
                self._check_and_finish(req)
            return True

        if self.kv_cache_manager is None:
            raise ValueError("cached execution requires a KV cache manager")

        decodable = []
        for req in active:
            if self.kv_cache_manager.allocate_request(req):
                decodable.append(req)

        if not decodable:
            return False

        active = decodable

        input_ids = torch.cat(
            [req.token_ids[:, -1:] for req in active],
            dim=0,
        )
        position_ids = torch.tensor(
            [[req.seq_len - 1] for req in active],
            dtype=torch.long,
            device=self.device,
        )
        batch_cache = BatchKVCache(
            [self.kv_cache_manager.get_cache(req) for req in active]
        )

        logits = self.model(
            input_ids,
            position_ids=position_ids,
            kv_cache=batch_cache,
        )  # [B, T, vocab]

        for row, req in enumerate(active):
            token = sample(
                logits[row : row + 1, -1, :],
                req.temperature,
                req.top_k,
                req.top_p,
            )  # [1, 1]
            req.append(token)
            self._check_and_finish(req)
        return True

    def _cleanup(self, use_cache: bool = True) -> bool:
        logger.debug(
            "Clean Up: running=%d, waiting=%d",
            len(self.scheduler.running),
            len(self.scheduler.waiting),
        )
        newly_finished = self.scheduler.evict_finished()
        for req in newly_finished:
            if use_cache and self.kv_cache_manager is not None:
                self.kv_cache_manager.free_request(req.id)
        return bool(newly_finished)

    def step(self, use_cache: bool = True) -> bool:
        self.step_count += 1
        logger.debug(
            "Step %s: running=%d, waiting=%d - device=%s",
            self.step_count,
            len(self.scheduler.running),
            len(self.scheduler.waiting),
            self.device,
        )
        progressed = False
        # 0. Check for new requests and prefill sequentially
        for req in self.scheduler.get_candidates():
            if use_cache:
                if self.kv_cache_manager is None:
                    raise ValueError("cached execution requires a KV cache manager")
                if not self.kv_cache_manager.allocate_request(req):
                    break
            self.scheduler.pop_request(req)
            self._prefill(req, use_cache)
            progressed = True
            self.scheduler.start_request(req)
            if use_cache and req.status == RequestStatus.FINISHED:
                if self.kv_cache_manager is None:
                    raise ValueError("cached execution requires a KV cache manager")
                self.kv_cache_manager.free_request(req.id)
        # 1. Decode all active requests
        progressed = self._decode(use_cache) or progressed
        # 2. Clean up finished requests
        progressed = self._cleanup(use_cache) or progressed
        return progressed

    def run(
        self,
        max_steps: int | None = None,
        timeout: float | None = None,
        *,
        use_cache: bool = True,
    ) -> None:
        if use_cache and self.kv_cache_manager is None:
            raise ValueError("cached execution requires a KV cache manager")

        start_time = time.time()
        step = 0
        while self.scheduler.has_work():
            if max_steps and step >= max_steps:
                break
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError("Engine run exceeded timeout")
            with torch.inference_mode():
                progressed = self.step(use_cache)
            if not progressed and self.scheduler.has_work():
                waiting = [req.id for req in self.scheduler.waiting]
                running = [req.id for req in self.scheduler.running]
                raise RuntimeError(
                    "engine made no progress; requests cannot currently be "
                    f"admitted or decoded (waiting={waiting}, running={running})"
                )
            step += 1
