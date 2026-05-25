"""Yahoo Finance market data via the `yfinance` package.

We use `yfinance.Ticker(symbol).history(...)` to fetch daily bars. yfinance
already handles split/dividend events by returning both raw OHLCV and the
adjusted close. It also returns the dividend amount and split ratio per bar
when the `actions=True` parameter is set.

Known limitations documented in the connector:

  - Survivorship bias: only currently-listed tickers return data. Delisted
    companies are silently empty.
  - Rate limit: yfinance scrapes Yahoo's web endpoints; no documented rate
    limit. Practical experience caps sustained throughput around 5 req/s
    before IP bans. We enforce a process-wide token bucket at 4 req/s
    (one token per `fetch_history` call) so a single job cannot exceed the
    safe rate regardless of caller concurrency. Cross-process coordination
    (multiple parallel jobs) is the orchestrator's responsibility — at most
    one Yahoo market job should be active at a time.
  - 429 / "Too Many Requests": yfinance swallows HTTP status, surfacing
    rate-limit responses as opaque exceptions. We retry on messages
    containing "429" or "Too Many Requests" with exponential backoff
    (max 3 attempts, base 2s).
  - Adjusted close is dividend- and split-adjusted; the unadjusted close is
    the raw `Close` column.
  - Date interval: yfinance's `end` parameter is exclusive, but
    `fetch_history(end=X)` here treats X as INCLUSIVE — we forward `end+1`
    internally so callers don't have to remember the convention.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterator, Optional

import pandas as pd
import yfinance as yf

from data.sources._throttle import TokenBucket

# 4 req/s with capacity=4 stays under Yahoo's ~5 req/s soft cap while permitting
# small bursts. Process-wide singleton — all callers share this budget.
_BUCKET = TokenBucket(rate=4.0, capacity=4.0)

_RATE_LIMIT_HINTS = ("429", "Too Many Requests", "rate limit", "Rate limit")
_MAX_ATTEMPTS = 3
_BACKOFF_BASE_S = 2.0


@dataclass(frozen=True)
class ParsedBar:
    bar_date: date
    open: float
    high: float
    low: float
    close: float
    adj_close: float
    volume: float
    dividend: float
    split_ratio: float


def _looks_like_rate_limit(exc: BaseException) -> bool:
    msg = str(exc)
    return any(h in msg for h in _RATE_LIMIT_HINTS)


def _history_with_retry(t: "yf.Ticker", kwargs: dict) -> pd.DataFrame:
    last_exc: Optional[BaseException] = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        _BUCKET.acquire()
        try:
            return t.history(**kwargs)
        except Exception as e:  # noqa: BLE001 — yfinance raises opaque errors
            last_exc = e
            if attempt == _MAX_ATTEMPTS or not _looks_like_rate_limit(e):
                raise
            time.sleep(_BACKOFF_BASE_S * (2 ** (attempt - 1)))
    # Defensive: loop always returns or raises.
    assert last_exc is not None
    raise last_exc


def fetch_history(
    ticker: str,
    start: Optional[date] = None,
    end: Optional[date] = None,
    *,
    interval: str = "1d",
) -> Iterator[ParsedBar]:
    """Stream OHLCV bars for `ticker` over [start, end].

    `interval` ∈ {'1d', '1wk', '1mo'}; the connector keeps the same Modality
    record shape regardless of interval. Quarterly aggregation is done
    downstream by the consumer (no need to re-fetch).
    """
    t = yf.Ticker(ticker)
    kwargs: dict = {"interval": interval, "actions": True, "auto_adjust": False}
    if start is not None:
        kwargs["start"] = start.isoformat()
    if end is not None:
        # yfinance's `end` is EXCLUSIVE — bars are returned strictly before
        # this date. Our connector contract is [start, end] inclusive, so we
        # bump by one calendar day before forwarding.
        kwargs["end"] = (end + timedelta(days=1)).isoformat()
    df: pd.DataFrame = _history_with_retry(t, kwargs)
    if df.empty:
        return
    for ts, row in df.iterrows():
        try:
            bar_date_val = ts.date()
        except AttributeError:
            bar_date_val = pd.Timestamp(ts).date()
        volume = float(row.get("Volume", 0) or 0)
        yield ParsedBar(
            bar_date=bar_date_val,
            open=float(row["Open"]),
            high=float(row["High"]),
            low=float(row["Low"]),
            close=float(row["Close"]),
            adj_close=float(row.get("Adj Close", row["Close"])),
            volume=volume,
            dividend=float(row.get("Dividends", 0) or 0),
            split_ratio=float(row.get("Stock Splits", 0) or 0) or 1.0,
        )
