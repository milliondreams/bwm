"""Market bar: one (entity, frequency, date) OHLCV observation.

PIT semantics:
  effective_date    = bar date (end of day for daily bars)
  availability_date = same as effective_date — end-of-day prices are
                      knowable to outsiders by close of that trading day.
  restated_at       = unused (market data is final once printed; corporate
                      actions are reflected in adj_close, not by mutating
                      prior bars).

Survivorship: yfinance returns data only for currently-listed tickers. Bars
for delisted companies are unavailable from this source. The connector
documents this; consumers that need survivorship-bias-free data should
supplement from a paid source (CRSP).
"""
from __future__ import annotations

from typing import ClassVar, Optional

from pydantic import Field

from data.schemas.pit import Modality, PITRecord


class MarketBar(PITRecord):
    """OHLCV bar for one entity at one frequency."""

    PARTITION_KEYS: ClassVar[list[str]] = ["frequency"]

    # yfinance scrapes Yahoo endpoints; very rarely returns transient bad data.
    # Slight discount on source_certainty vs first-party exchange data.
    DEFAULT_PERSPECTIVE: ClassVar[str] = "yahoo_finance"
    DEFAULT_SOURCE_CERTAINTY: ClassVar[float] = 0.95

    modality: Modality = Modality.MARKET

    frequency: str = Field(..., description="'daily', 'weekly', 'monthly', 'quarterly'")
    open: float
    high: float
    low: float
    close: float
    adj_close: float = Field(..., description="Split- and dividend-adjusted close")
    volume: float
    dividend: float = Field(default=0.0, description="Cash dividend per share on this bar")
    split_ratio: float = Field(
        default=1.0, description="Stock split ratio on this bar (1.0 = no split)"
    )
    ticker: str = Field(..., description="Ticker symbol used to fetch (provenance)")
