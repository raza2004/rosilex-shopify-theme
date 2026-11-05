"""Daily limit on image generations per visitor, by IP address (no accounts needed).

Every call to the image model counts, so a request that retries uses more than one. Calls are
reserved before generating and unused ones refunded, so parallel requests cannot go over the
limit. IP addresses are kept only as salted hashes, in memory, for the length of the window.
Counts are per process and start again when the service restarts.
"""

from __future__ import annotations

import hashlib
import math
import secrets
import threading
import time
from collections import defaultdict, deque


class DailyQuota:
    def __init__(self, limit: int, window_seconds: float = 24 * 3600, exempt=(), clock=time.time):
        self.limit = limit
        self.window = window_seconds
        self._exempt = {ip.strip() for ip in exempt if ip.strip()}
        self._clock = clock
        self._salt = secrets.token_bytes(16)
        self._calls: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def is_exempt(self, ip: str) -> bool:
        return ip in self._exempt

    def _key(self, ip: str) -> str:
        return hashlib.sha256(self._salt + ip.encode()).hexdigest()

    def _recent(self, key: str, now: float) -> deque[float]:
        calls = self._calls[key]
        while calls and calls[0] <= now - self.window:
            calls.popleft()
        if not calls:
            del self._calls[key]
            return self._calls[key]
        return calls

    def remaining(self, ip: str) -> int | None:
        """Calls left in the window, or None for an exempt IP (no limit)."""
        if self.is_exempt(ip):
            return None
        with self._lock:
            return max(0, self.limit - len(self._recent(self._key(ip), self._clock())))

    def reserve(self, ip: str, wanted: int) -> int | None:
        """Reserves up to wanted calls and returns how many were granted (None: exempt, no limit)."""
        if self.is_exempt(ip):
            return None
        with self._lock:
            now = self._clock()
            calls = self._recent(self._key(ip), now)
            granted = max(0, min(wanted, self.limit - len(calls)))
            calls.extend([now] * granted)
            return granted

    def refund(self, ip: str, count: int) -> None:
        """Gives back reserved calls that were not used."""
        if count <= 0 or self.is_exempt(ip):
            return
        with self._lock:
            calls = self._calls.get(self._key(ip))
            for _ in range(min(count, len(calls or ()))):
                calls.pop()

    def retry_after(self, ip: str) -> int:
        """Seconds until the oldest call in the window expires and a new one is allowed."""
        with self._lock:
            now = self._clock()
            calls = self._recent(self._key(ip), now)
            if len(calls) < self.limit:
                return 0
            return max(1, math.ceil(calls[0] + self.window - now))
