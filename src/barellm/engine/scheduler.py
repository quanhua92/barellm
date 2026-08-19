from collections import deque

from barellm.engine.request import Request, RequestStatus


class Scheduler:
    def __init__(self, max_batch=8):
        self.max_batch = max_batch
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.finished: list[Request] = []

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def get_candidates(self):
        slots = max(0, self.max_batch - len(self.running))
        return list(self.waiting)[:slots]

    def pop_request(self, request: Request) -> Request:
        if self.waiting and self.waiting[0] == request:
            return self.waiting.popleft()
        self.waiting.remove(request)
        return request

    def add_request(self, request: Request):
        self.waiting.append(request)

    def start_request(self, request: Request):
        if request.status == RequestStatus.FINISHED:
            self.finished.append(request)
        else:
            request.status = RequestStatus.RUNNING
            self.running.append(request)

    def evict_finished(self) -> list[Request]:
        newly = []
        still_running = []
        for req in self.running:
            if req.status == RequestStatus.FINISHED:
                self.finished.append(req)
                newly.append(req)
            else:
                still_running.append(req)
        self.running = still_running
        return newly
