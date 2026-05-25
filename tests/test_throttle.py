"""Unit tests for the thread-safe TokenBucket in data.sources._throttle."""
from __future__ import annotations

import threading
import time

import pytest

from data.sources._throttle import TokenBucket


def test_initial_burst_uses_capacity():
    # capacity=5 means the first 5 acquires should be near-instant.
    b = TokenBucket(rate=1.0, capacity=5.0)
    t0 = time.monotonic()
    for _ in range(5):
        b.acquire()
    assert time.monotonic() - t0 < 0.1


def test_refill_throttles_subsequent_calls():
    # rate=10/s, capacity=1: after the initial token, each subsequent acquire
    # waits ~0.1s. 4 extra tokens => ~0.4s total.
    b = TokenBucket(rate=10.0, capacity=1.0)
    t0 = time.monotonic()
    for _ in range(5):
        b.acquire()
    elapsed = time.monotonic() - t0
    assert 0.3 < elapsed < 0.7, f"expected ~0.4s, got {elapsed:.3f}s"


def test_concurrent_threads_share_budget():
    # rate=10/s capacity=1, 10 threads each acquire once. Total wall time
    # should be ~0.9s (first burst token + 9 refills at 0.1s each), not
    # ~0s (which would mean threads bypassed the lock).
    b = TokenBucket(rate=10.0, capacity=1.0)
    barrier = threading.Barrier(10)

    def worker():
        barrier.wait()
        b.acquire()

    threads = [threading.Thread(target=worker) for _ in range(10)]
    t0 = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.monotonic() - t0
    assert elapsed > 0.7, f"bucket did not serialize threads (elapsed={elapsed:.3f}s)"


def test_invalid_rate_raises():
    with pytest.raises(ValueError):
        TokenBucket(rate=0)
    with pytest.raises(ValueError):
        TokenBucket(rate=-1.0)
