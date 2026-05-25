"""Thread-safe rate-limiting primitives for synchronous data source connectors.

Companion to the asyncio-based `TokenBucket` in `edgar/client.py`. That one
guards aiohttp coroutines and uses `asyncio.Lock`; this one guards calls into
synchronous third-party libraries (`yfinance`, `requests`) invoked from worker
threads via `asyncio.to_thread`, and uses `threading.Lock` so it serializes
correctly across OS threads.

Place a module-level singleton `_BUCKET = TokenBucket(rate=..., capacity=...)`
in each source module and call `_BUCKET.acquire()` immediately before the
outbound network call. The bucket is process-wide, so all callers in a single
job share one budget regardless of how the CLI configures concurrency.

Cross-process coordination (multiple jobs simultaneously) is out of scope —
operator constraint enforced at the orchestrator layer (see `aml/modality_job.py`).
"""
from __future__ import annotations

import threading
import time
from typing import Optional


class TokenBucket:
    """Thread-safe token bucket. Steady refill at `rate` tokens/s up to `capacity`.

    Bursts above `rate` are permitted until the bucket drains, then `acquire()`
    blocks the calling thread just long enough for one token to refill.
    """

    def __init__(self, rate: float, capacity: Optional[float] = None) -> None:
        if rate <= 0:
            raise ValueError("rate must be positive")
        self.rate = rate
        self.capacity = capacity if capacity is not None else rate
        self._tokens = self.capacity
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, n: float = 1.0) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(
                    self.capacity, self._tokens + (now - self._last) * self.rate
                )
                self._last = now
                if self._tokens >= n:
                    self._tokens -= n
                    return
                deficit = n - self._tokens
                wait_s = deficit / self.rate
            time.sleep(wait_s)
