import logging
import time

import torch

from barellm.engine.batched_kv_cache import BatchKVCache
from barellm.engine.events import (
    AdmissionBlocked,
    CacheAllocated,
    CacheReleased,
    DecodeBatchEnd,
    DecodeBatchStart,
    EngineEvent,
    EngineStalled,
    EngineStepEnd,
    EngineStepStart,
    EventCallback,
    ModelForwardEnd,
    ModelForwardStart,
    PrefillEnd,
    PrefillStart,
    RequestAdmitted,
    RequestFinished,
    RequestSubmitted,
    SamplingEnd,
    SamplingStart,
    TimingConfig,
    TokenGenerated,
)
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

    def _emit(self, callback: EventCallback | None, event: EngineEvent) -> None:
        if callback is not None:
            callback(event)

    def _synchronize(self, timing: TimingConfig) -> None:
        if not timing.synchronize_device:
            return
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps" and torch.backends.mps.is_available():
            torch.mps.synchronize()

    def _phase_start(self, timing: TimingConfig) -> float:
        self._synchronize(timing)
        return time.perf_counter()

    def _phase_end(self, started_at: float, timing: TimingConfig) -> float:
        self._synchronize(timing)
        return time.perf_counter() - started_at

    def _emit_cache_allocated(
        self,
        req: Request,
        callback: EventCallback | None,
        block_ids: tuple[int, ...],
    ) -> None:
        if self.kv_cache_manager is None or not block_ids:
            return
        self._emit(
            callback,
            CacheAllocated(
                timestamp=time.perf_counter(),
                step=self.step_count,
                request_id=req.id,
                block_ids=block_ids,
                sequence_length=req.seq_len,
            ),
        )

    def _allocate_request(
        self,
        req: Request,
        callback: EventCallback | None,
    ) -> bool:
        """Allocate newly needed blocks and emit only real allocations."""
        if self.kv_cache_manager is None:
            raise ValueError("cached execution requires a KV cache manager")
        existing_blocks = self.kv_cache_manager.request_id_to_blocks.get(req.id, [])
        existing_count = len(existing_blocks)
        if not self.kv_cache_manager.allocate_request(req):
            return False
        current_blocks = self.kv_cache_manager.request_id_to_blocks.get(req.id, [])
        new_block_ids = tuple(
            block.block_id for block in current_blocks[existing_count:]
        )
        self._emit_cache_allocated(req, callback, new_block_ids)
        return True

    def _free_cache(
        self,
        request_id: str,
        callback: EventCallback | None,
    ) -> None:
        if self.kv_cache_manager is None:
            return
        blocks = self.kv_cache_manager.request_id_to_blocks.get(request_id, [])
        block_ids = tuple(block.block_id for block in blocks)
        self.kv_cache_manager.free_request(request_id)
        self._emit(
            callback,
            CacheReleased(
                timestamp=time.perf_counter(),
                step=self.step_count,
                request_id=request_id,
                block_ids=block_ids,
            ),
        )

    def _finish_request(
        self,
        req: Request,
        callback: EventCallback | None,
    ) -> None:
        if req.finish_event_emitted:
            return
        req.finish_event_emitted = True
        self._emit(
            callback,
            RequestFinished(
                timestamp=time.perf_counter(),
                step=self.step_count,
                request_id=req.id,
                finish_reason=req.finish_reason or FINISH_LENGTH,
                stop_reason=req.stop_reason,
                generated_count=req.generated_count,
                prompt_tokens=req.seq_len - req.generated_count,
                sequence_length=req.seq_len,
            ),
        )

    def _check_and_finish(
        self,
        req: Request,
        callback: EventCallback | None,
    ) -> None:
        """Check stops, emit the token event, then emit one finish event."""
        result = req.check_stop()
        if result is not None:
            req.finish_reason, req.stop_reason = result
            req.status = RequestStatus.FINISHED

        token_id = int(req.token_ids[0, -1].item())
        self._emit(
            callback,
            TokenGenerated(
                timestamp=time.perf_counter(),
                step=self.step_count,
                request_id=req.id,
                token_id=token_id,
                generated_count=req.generated_count,
                sequence_length=req.seq_len,
                is_first_token=req.generated_count == 1,
            ),
        )

        if req.on_token is not None:
            should_continue = req.on_token(token_id, req.seq_len)
            if should_continue is False:
                req.status = RequestStatus.FINISHED
                req.stop_reason = None
                req.finish_reason = FINISH_ABORT

        if req.status == RequestStatus.FINISHED:
            self._finish_request(req, callback)

    def _prefill(
        self,
        req: Request,
        use_cache: bool,
        callback: EventCallback | None,
        timing: TimingConfig,
    ) -> bool:
        logger.debug("Prefill Request: %s", req.id)
        prompt_tokens = req.seq_len
        self._emit(
            callback,
            PrefillStart(
                timestamp=time.perf_counter(),
                step=self.step_count,
                request_id=req.id,
                prompt_tokens=prompt_tokens,
                use_cache=use_cache,
            ),
        )
        started_at = self._phase_start(timing)

        if req.max_new_tokens <= 0:
            duration = self._phase_end(started_at, timing)
            self._emit(
                callback,
                PrefillEnd(
                    timestamp=time.perf_counter(),
                    step=self.step_count,
                    request_id=req.id,
                    prompt_tokens=prompt_tokens,
                    use_cache=use_cache,
                    duration_seconds=duration,
                ),
            )
            req.finish_reason = FINISH_LENGTH
            req.status = RequestStatus.FINISHED
            self._finish_request(req, callback)
            return True

        req.token_ids = req.token_ids.to(self.device)
        position_ids = torch.arange(
            req.seq_len, dtype=torch.long, device=self.device
        ).unsqueeze(0)

        cache = None
        if use_cache:
            if self.kv_cache_manager is None:
                raise ValueError("cached execution requires a KV cache manager")
            cache = self.kv_cache_manager.get_cache(req)

        request_ids = (req.id,)
        self._emit(
            callback,
            ModelForwardStart(
                timestamp=time.perf_counter(),
                step=self.step_count,
                request_id=req.id,
                phase="prefill",
                request_ids=request_ids,
                batch_size=1,
                input_tokens=req.seq_len,
                sequence_lengths=(req.seq_len,),
                use_cache=use_cache,
            ),
        )
        model_started_at = self._phase_start(timing)
        logits = self.model(
            req.token_ids,
            position_ids=position_ids,
            kv_cache=cache,
        )
        model_duration = self._phase_end(model_started_at, timing)
        self._emit(
            callback,
            ModelForwardEnd(
                timestamp=time.perf_counter(),
                step=self.step_count,
                request_id=req.id,
                phase="prefill",
                request_ids=request_ids,
                duration_seconds=model_duration,
            ),
        )
        self._emit(
            callback,
            SamplingStart(
                timestamp=time.perf_counter(),
                step=self.step_count,
                request_id=req.id,
                phase="prefill",
                request_ids=request_ids,
                batch_size=1,
            ),
        )
        sampling_started_at = self._phase_start(timing)
        next_token = sample(logits[:, -1, :], req.temperature, req.top_k, req.top_p)
        sampling_duration = self._phase_end(sampling_started_at, timing)
        self._emit(
            callback,
            SamplingEnd(
                timestamp=time.perf_counter(),
                step=self.step_count,
                request_id=req.id,
                phase="prefill",
                request_ids=request_ids,
                batch_size=1,
                duration_seconds=sampling_duration,
            ),
        )
        duration = self._phase_end(started_at, timing)
        self._emit(
            callback,
            PrefillEnd(
                timestamp=time.perf_counter(),
                step=self.step_count,
                request_id=req.id,
                prompt_tokens=prompt_tokens,
                use_cache=use_cache,
                duration_seconds=duration,
            ),
        )
        req.append(next_token)
        self._check_and_finish(req, callback)
        return True

    def _decode(
        self,
        use_cache: bool = True,
        callback: EventCallback | None = None,
        timing: TimingConfig | None = None,
    ) -> bool:
        logger.debug(
            "Decode: running=%d, waiting=%d",
            len(self.scheduler.running),
            len(self.scheduler.waiting),
        )
        timing = timing or TimingConfig()
        active = self.scheduler.running
        if not active:
            return False

        if not use_cache:
            for req in active:
                request_ids = (req.id,)
                self._emit(
                    callback,
                    DecodeBatchStart(
                        timestamp=time.perf_counter(),
                        step=self.step_count,
                        request_ids=request_ids,
                        use_cache=False,
                    ),
                )
                started_at = self._phase_start(timing)
                position_ids = torch.arange(
                    req.seq_len,
                    dtype=torch.long,
                    device=self.device,
                ).unsqueeze(0)
                self._emit(
                    callback,
                    ModelForwardStart(
                        timestamp=time.perf_counter(),
                        step=self.step_count,
                        request_id=req.id,
                        phase="decode",
                        request_ids=request_ids,
                        batch_size=1,
                        input_tokens=1,
                        sequence_lengths=(req.seq_len,),
                        use_cache=False,
                    ),
                )
                model_started_at = self._phase_start(timing)
                logits = self.model(
                    req.token_ids,
                    position_ids=position_ids,
                    kv_cache=None,
                )
                model_duration = self._phase_end(model_started_at, timing)
                self._emit(
                    callback,
                    ModelForwardEnd(
                        timestamp=time.perf_counter(),
                        step=self.step_count,
                        request_id=req.id,
                        phase="decode",
                        request_ids=request_ids,
                        duration_seconds=model_duration,
                    ),
                )
                self._emit(
                    callback,
                    SamplingStart(
                        timestamp=time.perf_counter(),
                        step=self.step_count,
                        request_id=req.id,
                        phase="decode",
                        request_ids=request_ids,
                        batch_size=1,
                    ),
                )
                sampling_started_at = self._phase_start(timing)
                token = sample(
                    logits[:, -1, :],
                    req.temperature,
                    req.top_k,
                    req.top_p,
                )
                sampling_duration = self._phase_end(sampling_started_at, timing)
                self._emit(
                    callback,
                    SamplingEnd(
                        timestamp=time.perf_counter(),
                        step=self.step_count,
                        request_id=req.id,
                        phase="decode",
                        request_ids=request_ids,
                        batch_size=1,
                        duration_seconds=sampling_duration,
                    ),
                )
                duration = self._phase_end(started_at, timing)
                self._emit(
                    callback,
                    DecodeBatchEnd(
                        timestamp=time.perf_counter(),
                        step=self.step_count,
                        request_ids=request_ids,
                        use_cache=False,
                        duration_seconds=duration,
                    ),
                )
                req.append(token)
                self._check_and_finish(req, callback)
            return True

        if self.kv_cache_manager is None:
            raise ValueError("cached execution requires a KV cache manager")

        decodable = []
        for req in active:
            if self._allocate_request(req, callback):
                decodable.append(req)
            else:
                self._emit(
                    callback,
                    AdmissionBlocked(
                        timestamp=time.perf_counter(),
                        step=self.step_count,
                        request_id=req.id,
                        sequence_length=req.seq_len,
                        reason="decode cache capacity",
                    ),
                )

        if not decodable:
            return False

        active = decodable
        request_ids = tuple(req.id for req in active)
        self._emit(
            callback,
            DecodeBatchStart(
                timestamp=time.perf_counter(),
                step=self.step_count,
                request_ids=request_ids,
                use_cache=True,
            ),
        )
        started_at = self._phase_start(timing)
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
        self._emit(
            callback,
            ModelForwardStart(
                timestamp=time.perf_counter(),
                step=self.step_count,
                phase="decode",
                request_ids=request_ids,
                batch_size=len(active),
                input_tokens=1,
                sequence_lengths=tuple(req.seq_len for req in active),
                use_cache=True,
            ),
        )
        model_started_at = self._phase_start(timing)
        logits = self.model(
            input_ids,
            position_ids=position_ids,
            kv_cache=batch_cache,
        )
        model_duration = self._phase_end(model_started_at, timing)
        self._emit(
            callback,
            ModelForwardEnd(
                timestamp=time.perf_counter(),
                step=self.step_count,
                phase="decode",
                request_ids=request_ids,
                duration_seconds=model_duration,
            ),
        )
        self._emit(
            callback,
            SamplingStart(
                timestamp=time.perf_counter(),
                step=self.step_count,
                phase="decode",
                request_ids=request_ids,
                batch_size=len(active),
            ),
        )
        sampling_started_at = self._phase_start(timing)
        tokens = [
            sample(
                logits[row : row + 1, -1, :],
                req.temperature,
                req.top_k,
                req.top_p,
            )
            for row, req in enumerate(active)
        ]
        sampling_duration = self._phase_end(sampling_started_at, timing)
        self._emit(
            callback,
            SamplingEnd(
                timestamp=time.perf_counter(),
                step=self.step_count,
                phase="decode",
                request_ids=request_ids,
                batch_size=len(active),
                duration_seconds=sampling_duration,
            ),
        )
        duration = self._phase_end(started_at, timing)
        self._emit(
            callback,
            DecodeBatchEnd(
                timestamp=time.perf_counter(),
                step=self.step_count,
                request_ids=request_ids,
                use_cache=True,
                duration_seconds=duration,
            ),
        )
        for req, token in zip(active, tokens):
            req.append(token)
            self._check_and_finish(req, callback)
        return True

    def _cleanup(
        self,
        use_cache: bool = True,
        callback: EventCallback | None = None,
    ) -> bool:
        logger.debug(
            "Clean Up: running=%d, waiting=%d",
            len(self.scheduler.running),
            len(self.scheduler.waiting),
        )
        newly_finished = self.scheduler.evict_finished()
        for req in newly_finished:
            if use_cache and self.kv_cache_manager is not None:
                self._free_cache(req.id, callback)
        return bool(newly_finished)

    def step(
        self,
        use_cache: bool = True,
        callback: EventCallback | None = None,
        timing: TimingConfig | None = None,
    ) -> bool:
        timing = timing or TimingConfig()
        self.step_count += 1
        step = self.step_count
        self._emit(
            callback,
            EngineStepStart(
                timestamp=time.perf_counter(),
                step=step,
            ),
        )
        progressed = False
        for req in self.scheduler.get_candidates():
            if use_cache:
                if self.kv_cache_manager is None:
                    raise ValueError("cached execution requires a KV cache manager")
                if not self._allocate_request(req, callback):
                    self._emit(
                        callback,
                        AdmissionBlocked(
                            timestamp=time.perf_counter(),
                            step=step,
                            request_id=req.id,
                            sequence_length=req.seq_len,
                            reason="prefill cache capacity",
                        ),
                    )
                    break
            self.scheduler.pop_request(req)
            self._emit(
                callback,
                RequestAdmitted(
                    timestamp=time.perf_counter(),
                    step=step,
                    request_id=req.id,
                    prompt_tokens=req.seq_len,
                    use_cache=use_cache,
                ),
            )
            self._prefill(req, use_cache, callback, timing)
            progressed = True
            self.scheduler.start_request(req)
            if use_cache and req.status == RequestStatus.FINISHED:
                self._free_cache(req.id, callback)
        progressed = self._decode(use_cache, callback, timing) or progressed
        progressed = self._cleanup(use_cache, callback) or progressed
        self._emit(
            callback,
            EngineStepEnd(
                timestamp=time.perf_counter(),
                step=step,
                progressed=progressed,
            ),
        )
        return progressed

    def run(
        self,
        max_steps: int | None = None,
        timeout: float | None = None,
        *,
        use_cache: bool = True,
        on_event: EventCallback | None = None,
        timing: TimingConfig | None = None,
    ) -> None:
        if use_cache and self.kv_cache_manager is None:
            raise ValueError("cached execution requires a KV cache manager")
        timing = timing or TimingConfig()
        for req in self.scheduler.waiting:
            if req.submitted_event_emitted:
                continue
            req.submitted_event_emitted = True
            self._emit(
                on_event,
                RequestSubmitted(
                    timestamp=time.perf_counter(),
                    step=self.step_count,
                    request_id=req.id,
                    prompt_tokens=req.seq_len,
                ),
            )

        start_time = time.time()
        steps = 0
        while self.scheduler.has_work():
            if max_steps and steps >= max_steps:
                break
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError("Engine run exceeded timeout")
            with torch.inference_mode():
                progressed = self.step(use_cache, on_event, timing)
            if not progressed and self.scheduler.has_work():
                waiting = tuple(req.id for req in self.scheduler.waiting)
                running = tuple(req.id for req in self.scheduler.running)
                reason = (
                    "engine made no progress; requests cannot currently be "
                    f"admitted or decoded (waiting={list(waiting)}, "
                    f"running={list(running)})"
                )
                self._emit(
                    on_event,
                    EngineStalled(
                        timestamp=time.perf_counter(),
                        step=self.step_count,
                        waiting_ids=waiting,
                        running_ids=running,
                        reason=reason,
                    ),
                )
                raise RuntimeError(reason)
            steps += 1
