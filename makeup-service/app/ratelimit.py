"""A basic in-memory sliding window limiter. Per process: each replica keeps its own counts."""

from __future__ import annotations

import math
import threading
import time
from collections import defaultdict, deque


class RateLimiter:
    def __init__(self, limit: int, window_seconds: float, clock=time.monotonic):
        self.limit = limit
        self.window = window_seconds
        self._clock = clock
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> int | None:
        """Records a hit. Returns None if allowed, otherwise seconds until the next request is allowed."""
        now = self._clock()
        with self._lock:
            hits = self._hits[key]
            while hits and hits[0] <= now - self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                return max(1, math.ceil(hits[0] + self.window - now))
            hits.append(now)
            if len(self._hits) > 10_000:  # drop idle clients so memory stays bounded
                for stale in [k for k, v in self._hits.items() if not v]:
                    del self._hits[stale]
            return None
