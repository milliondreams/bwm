"""Verify uspto.fetch_patents_by_assignee throttles, retries on 429/503,
and honors Retry-After.
"""
from __future__ import annotations

import time
from datetime import date

import pytest

from data.sources.patents import uspto


class _FakeResponse:
    def __init__(self, status_code: int, json_body: dict | None = None,
                 retry_after: str | None = None):
        self.status_code = status_code
        self._body = json_body or {}
        self.headers = {}
        if retry_after is not None:
            self.headers["Retry-After"] = retry_after

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


def _swap_bucket(monkeypatch, rate=100.0, capacity=10.0):
    from data.sources._throttle import TokenBucket
    monkeypatch.setattr(uspto, "_BUCKET", TokenBucket(rate=rate, capacity=capacity))


def test_retry_on_429_then_success(monkeypatch):
    _swap_bucket(monkeypatch)
    calls = {"n": 0}

    def fake_post(url, json, headers, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            return _FakeResponse(429, retry_after="0")
        return _FakeResponse(200, json_body={"patents": []})

    monkeypatch.setattr(uspto.requests, "post", fake_post)
    list(uspto.fetch_patents_by_assignee("Acme"))
    assert calls["n"] == 2


def test_retry_after_is_honored(monkeypatch):
    _swap_bucket(monkeypatch)
    calls = {"n": 0}

    def fake_post(url, json, headers, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            return _FakeResponse(503, retry_after="0.3")
        return _FakeResponse(200, json_body={"patents": []})

    monkeypatch.setattr(uspto.requests, "post", fake_post)
    t0 = time.monotonic()
    list(uspto.fetch_patents_by_assignee("Acme"))
    elapsed = time.monotonic() - t0
    # Retry-After=0.3 plus tenacity backoff (≥1.0s initial). Just confirm
    # we slept at least the Retry-After window.
    assert elapsed >= 0.3


def test_giveup_after_repeated_429(monkeypatch):
    _swap_bucket(monkeypatch)
    calls = {"n": 0}

    def fake_post(url, json, headers, timeout):
        calls["n"] += 1
        return _FakeResponse(429, retry_after="0")

    monkeypatch.setattr(uspto.requests, "post", fake_post)
    with pytest.raises(uspto._RetryableHTTPError):
        list(uspto.fetch_patents_by_assignee("Acme"))
    assert calls["n"] == 4  # stop_after_attempt(4)


def test_bucket_throttles_pagination(monkeypatch):
    """Pagination loop should pay the bucket cost per page."""
    from data.sources._throttle import TokenBucket
    monkeypatch.setattr(uspto, "_BUCKET", TokenBucket(rate=20.0, capacity=1.0))

    # 3 pages of 100, then empty -> 3 acquires total (rate-bound).
    pages = [
        {"patents": [{"patent_number": str(i),
                      "patent_date": "2020-01-01"} for i in range(100)]},
        {"patents": [{"patent_number": str(i),
                      "patent_date": "2020-01-01"} for i in range(100)]},
        {"patents": []},
    ]
    idx = {"i": 0}

    def fake_post(url, json, headers, timeout):
        body = pages[idx["i"]]
        idx["i"] += 1
        return _FakeResponse(200, json_body=body)

    monkeypatch.setattr(uspto.requests, "post", fake_post)
    t0 = time.monotonic()
    list(uspto.fetch_patents_by_assignee("Acme"))
    elapsed = time.monotonic() - t0
    # 3 acquires at 20/s capacity=1 => first instant, two more at 0.05s = ~0.1s.
    assert 0.05 < elapsed < 0.5
