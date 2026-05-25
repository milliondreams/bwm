"""Verify yahoo.fetch_history enforces the module-level token bucket and
retries on rate-limit-looking exceptions.
"""
from __future__ import annotations

import time
from datetime import date

import pandas as pd
import pytest

from data.sources.market import yahoo


class _StubTicker:
    """Counts calls to .history(); returns an empty DataFrame so fetch_history
    yields nothing (we only care about call rate, not parse output)."""

    calls = 0

    def __init__(self, *_a, **_kw):
        pass

    def history(self, **_kw) -> pd.DataFrame:
        _StubTicker.calls += 1
        return pd.DataFrame()


def _swap_bucket(monkeypatch, rate: float, capacity: float):
    """Replace the module-level bucket with a fresh, fast one for tests."""
    from data.sources._throttle import TokenBucket

    monkeypatch.setattr(yahoo, "_BUCKET", TokenBucket(rate=rate, capacity=capacity))


def test_bucket_throttles_sequential_calls(monkeypatch):
    monkeypatch.setattr(yahoo.yf, "Ticker", _StubTicker)
    _swap_bucket(monkeypatch, rate=20.0, capacity=1.0)
    _StubTicker.calls = 0

    t0 = time.monotonic()
    for _ in range(5):
        list(yahoo.fetch_history("FAKE", start=date(2020, 1, 1), end=date(2020, 1, 2)))
    elapsed = time.monotonic() - t0

    assert _StubTicker.calls == 5
    # 5 calls at 20/s with capacity 1: first instant, 4 refills at 0.05s = 0.2s.
    assert 0.15 < elapsed < 0.5, f"expected ~0.2s, got {elapsed:.3f}s"


class _RateLimitedOnceTicker:
    """Raises a '429' Exception on the first call, returns empty df after."""

    calls = 0

    def __init__(self, *_a, **_kw):
        pass

    def history(self, **_kw) -> pd.DataFrame:
        _RateLimitedOnceTicker.calls += 1
        if _RateLimitedOnceTicker.calls == 1:
            raise Exception("HTTP 429 Too Many Requests")
        return pd.DataFrame()


def test_retry_on_rate_limit_exception(monkeypatch):
    monkeypatch.setattr(yahoo.yf, "Ticker", _RateLimitedOnceTicker)
    _swap_bucket(monkeypatch, rate=100.0, capacity=10.0)
    # Skip the real 2s backoff in tests.
    monkeypatch.setattr(yahoo, "_BACKOFF_BASE_S", 0.01)
    _RateLimitedOnceTicker.calls = 0

    list(yahoo.fetch_history("FAKE", start=date(2020, 1, 1), end=date(2020, 1, 2)))
    assert _RateLimitedOnceTicker.calls == 2  # one failure, one successful retry


class _NonRetryableTicker:
    calls = 0

    def __init__(self, *_a, **_kw):
        pass

    def history(self, **_kw):
        _NonRetryableTicker.calls += 1
        raise ValueError("ticker symbol not found")


def test_non_rate_limit_exception_does_not_retry(monkeypatch):
    monkeypatch.setattr(yahoo.yf, "Ticker", _NonRetryableTicker)
    _swap_bucket(monkeypatch, rate=100.0, capacity=10.0)
    _NonRetryableTicker.calls = 0

    with pytest.raises(ValueError):
        list(yahoo.fetch_history("FAKE", start=date(2020, 1, 1), end=date(2020, 1, 2)))
    assert _NonRetryableTicker.calls == 1
