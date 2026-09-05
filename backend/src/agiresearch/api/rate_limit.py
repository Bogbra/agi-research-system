"""A minimal in-process token-bucket rate limiter.

Deliberately not a distributed limiter (Redis, API gateway) — those are the
right tool for a multi-instance deployment and are a one-line swap at this
call site (`RateLimitMiddleware.__init__`) once the service actually scales
horizontally. What this *does* demonstrate: an unmetered endpoint that can
trigger several LLM calls per request is a cost and abuse vector, so
nothing ships without a limiter in front of it, even a simple one.
"""

from __future__ import annotations

import time
from collections import defaultdict
from threading import Lock

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


class _TokenBucket:
    __slots__ = ("tokens", "last_refill")

    def __init__(self, capacity: float) -> None:
        self.tokens = capacity
        self.last_refill = time.monotonic()


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, requests_per_minute: int = 60, burst: int | None = None) -> None:
        super().__init__(app)
        self._rate = requests_per_minute / 60.0
        self._capacity = float(burst or requests_per_minute)
        self._buckets: dict[str, _TokenBucket] = defaultdict(lambda: _TokenBucket(self._capacity))
        self._lock = Lock()

    def _client_key(self, request: Request) -> str:
        auth = request.headers.get("authorization", "")
        if auth:
            return auth
        client = request.client
        return client.host if client else "unknown"

    async def dispatch(self, request: Request, call_next):
        key = self._client_key(request)
        with self._lock:
            bucket = self._buckets[key]
            now = time.monotonic()
            elapsed = now - bucket.last_refill
            bucket.tokens = min(self._capacity, bucket.tokens + elapsed * self._rate)
            bucket.last_refill = now

            if bucket.tokens < 1:
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded. Please retry shortly."},
                )
            bucket.tokens -= 1

        return await call_next(request)
