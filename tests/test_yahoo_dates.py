"""Verify yahoo.fetch_history treats `end` as inclusive by forwarding end+1
to yfinance (which uses an exclusive end internally).
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from data.sources.market import yahoo


class _CapturingTicker:
    last_kwargs: dict | None = None

    def __init__(self, *_a, **_kw):
        pass

    def history(self, **kwargs) -> pd.DataFrame:
        _CapturingTicker.last_kwargs = kwargs
        return pd.DataFrame()


def _swap_bucket(monkeypatch):
    from data.sources._throttle import TokenBucket
    monkeypatch.setattr(yahoo, "_BUCKET", TokenBucket(rate=100.0, capacity=10.0))


def test_end_is_passed_as_end_plus_one(monkeypatch):
    monkeypatch.setattr(yahoo.yf, "Ticker", _CapturingTicker)
    _swap_bucket(monkeypatch)
    _CapturingTicker.last_kwargs = None

    list(yahoo.fetch_history("FAKE", start=date(2024, 1, 1), end=date(2024, 1, 31)))

    assert _CapturingTicker.last_kwargs is not None
    assert _CapturingTicker.last_kwargs["start"] == "2024-01-01"
    assert _CapturingTicker.last_kwargs["end"] == "2024-02-01"  # end + 1 day


def test_end_omitted_when_not_supplied(monkeypatch):
    monkeypatch.setattr(yahoo.yf, "Ticker", _CapturingTicker)
    _swap_bucket(monkeypatch)
    _CapturingTicker.last_kwargs = None

    list(yahoo.fetch_history("FAKE", start=date(2024, 1, 1)))

    assert "end" not in _CapturingTicker.last_kwargs
