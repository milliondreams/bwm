"""Yahoo Finance market data via the `yfinance` package.

We use `yfinance.Ticker(symbol).history(...)` to fetch daily bars. yfinance
already handles split/dividend events by returning both raw OHLCV and the
adjusted close. It also returns the dividend amount and split ratio per bar
when the `actions=True` parameter is set.

Known limitations documented in the connector:

  - Survivorship bias: only currently-listed tickers return data. Delisted
    companies are silently empty.
  - Rate limit: yfinance scrapes Yahoo's web endpoints; no documented rate
    limit. Practically we throttle to ~5 req/s to avoid IP bans.
  - Adjusted close is dividend- and split-adjusted; the unadjusted close is
    the raw `Close` column.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterator, Optional

import pandas as pd
import yfinance as yf


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
        kwargs["end"] = end.isoformat()
    df: pd.DataFrame = t.history(**kwargs)
    if df.empty:
        return
    # yfinance returns a DatetimeIndex; convert and iterate.
    for ts, row in df.iterrows():
        # ts is a Timestamp (tz-aware for some intervals). We take the .date()
        # — the bar identifies a calendar date, not an instant.
        try:
            bar_date_val = ts.date()
        except AttributeError:
            bar_date_val = pd.Timestamp(ts).date()
        # yfinance occasionally returns NaN volume for some pre-IPO rows.
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
