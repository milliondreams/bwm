"""PIT-bounded entity snapshots — the only safe way training code touches canonical.

The BWM spec's FR-2 says training and inference must never use information not
knowable on the prediction date. Concretely, when constructing a training
example for entity E at time t (predicting the regime at t+h), the *inputs*
to the encoder must use the data that an outside observer could have seen
*on date t*, not the latest-restated values we have now. A model trained
on restated values silently "knows the answer" and every accuracy number
it produces is fictional.

This module provides the primitive that enforces that invariant:

    snapshot = pit.snapshot(entity_id, as_of=date(2018, 6, 30))
    # snapshot.financials is bounded by availability_date <= 2018-06-30
    # snapshot.insider_trades, etc. — same bound

`EntitySnapshot` is `frozen=True`, so once constructed it can't accidentally
acquire newer rows. Its `__post_init__` runs cheap PIT assertions on every
modality DataFrame, so if a caller tries to construct one by hand with bad
data, Python raises immediately. The supported way to create a snapshot is
`PITEngine.snapshot(...)`; the dataclass constructor is essentially private.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Iterable, Optional

import pandas as pd

from data.schemas.filing_text import FilingTextChunk
from data.schemas.financial import FinancialFact
from data.schemas.insider_trade import InsiderTrade
from data.schemas.pit import Modality
from data.schemas.supply_chain import SupplyChainEdge


def lazy_schema_cls(modality: Modality):
    """Return the concrete PITRecord subclass for `modality`, importing lazily.

    Used by `PITEngine.write` to look up v3 PIT defaults (and PARTITION_KEYS)
    without coupling the engine to every schema module. Returns None if no
    schema is known for the modality."""
    if modality is Modality.FINANCIALS:
        return FinancialFact
    if modality is Modality.INSIDER_TRADES:
        return InsiderTrade
    if modality is Modality.FILINGS_TEXT:
        return FilingTextChunk
    if modality is Modality.SUPPLY_CHAIN:
        return SupplyChainEdge
    phase_a2_imports = {
        Modality.MARKET: ("data.schemas.market_bar", "MarketBar"),
        Modality.MACRO: ("data.schemas.macro_observation", "MacroObservation"),
        Modality.NEWS: ("data.schemas.news_event", "NewsEvent"),
        Modality.PATENTS: ("data.schemas.patent", "Patent"),
        Modality.EARNINGS_CALLS: ("data.schemas.earnings_call", "EarningsCallExcerpt"),
        Modality.HIRING: ("data.schemas.hiring_observation", "HiringObservation"),
    }
    if modality in phase_a2_imports:
        mod_name, cls_name = phase_a2_imports[modality]
        try:
            mod = __import__(mod_name, fromlist=[cls_name])
            return getattr(mod, cls_name)
        except (ImportError, AttributeError):
            return None
    return None


def _lazy_partition_keys(modality: Modality) -> list[str]:
    """Return PARTITION_KEYS for a modality, importing the schema lazily so
    snapshot.py doesn't hard-import every modality (some are added late)."""
    # Always-present modalities (the original four).
    if modality is Modality.FINANCIALS:
        return FinancialFact.PARTITION_KEYS
    if modality is Modality.INSIDER_TRADES:
        return InsiderTrade.PARTITION_KEYS
    if modality is Modality.FILINGS_TEXT:
        return FilingTextChunk.PARTITION_KEYS
    if modality is Modality.SUPPLY_CHAIN:
        return SupplyChainEdge.PARTITION_KEYS
    # Phase A.2 modalities — imported lazily and *defensively*: each is added
    # in its own stage, and a missing schema simply means "this modality has
    # no canonical data yet." Falling back to [] is safe because the engine's
    # column-inference path handles partition_keys=[] (no dedup keys beyond
    # entity_id+effective_date+source_ref).
    phase_a2_imports = {
        Modality.MARKET: ("data.schemas.market_bar", "MarketBar"),
        Modality.MACRO: ("data.schemas.macro_observation", "MacroObservation"),
        Modality.NEWS: ("data.schemas.news_event", "NewsEvent"),
        Modality.PATENTS: ("data.schemas.patent", "Patent"),
        Modality.EARNINGS_CALLS: ("data.schemas.earnings_call", "EarningsCallExcerpt"),
        Modality.HIRING: ("data.schemas.hiring_observation", "HiringObservation"),
    }
    if modality in phase_a2_imports:
        mod_name, cls_name = phase_a2_imports[modality]
        try:
            mod = __import__(mod_name, fromlist=[cls_name])
            return getattr(mod, cls_name).PARTITION_KEYS
        except (ImportError, AttributeError):
            return []
    return []


@dataclass(frozen=True)
class EntitySnapshot:
    """Frozen, PIT-bounded view of one entity's data at a given as_of date.

    Invariants asserted at construction time:
      • Every non-empty modality DataFrame has all rows with
        `availability_date <= as_of`.
      • Every non-empty modality DataFrame has no row whose `restated_at`
        is non-null *and* <= as_of (those rows have been superseded by
        a more recent filing that was already published as of `as_of`).

    A snapshot is the contract that the training data loader and the
    inference engine share with the PIT layer: nothing in this snapshot
    "knows" anything that wasn't public on `as_of`.

    Carries 10 modality DataFrames matching spec §5.6: the 4 EDGAR-derived
    modalities plus market, macro, news, patents, earnings_calls, hiring.
    """

    entity_id: str
    as_of: date
    # EDGAR-derived (Stages 2-6)
    financials: pd.DataFrame = field(default_factory=pd.DataFrame)
    insider_trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    filings_text: pd.DataFrame = field(default_factory=pd.DataFrame)
    supply_chain: pd.DataFrame = field(default_factory=pd.DataFrame)
    # Phase A.2 modalities
    market: pd.DataFrame = field(default_factory=pd.DataFrame)
    macro: pd.DataFrame = field(default_factory=pd.DataFrame)
    news: pd.DataFrame = field(default_factory=pd.DataFrame)
    patents: pd.DataFrame = field(default_factory=pd.DataFrame)
    earnings_calls: pd.DataFrame = field(default_factory=pd.DataFrame)
    hiring: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __post_init__(self) -> None:
        for name, df in self._all_frames():
            _assert_pit_safe(name, df, self.as_of)

    def _all_frames(self) -> list[tuple[str, pd.DataFrame]]:
        return [
            ("financials", self.financials),
            ("insider_trades", self.insider_trades),
            ("filings_text", self.filings_text),
            ("supply_chain", self.supply_chain),
            ("market", self.market),
            ("macro", self.macro),
            ("news", self.news),
            ("patents", self.patents),
            ("earnings_calls", self.earnings_calls),
            ("hiring", self.hiring),
        ]

    # --- read helpers ---------------------------------------------------

    def get(self, modality: Modality) -> pd.DataFrame:
        """Return the bounded DataFrame for one modality."""
        return {
            Modality.FINANCIALS: self.financials,
            Modality.INSIDER_TRADES: self.insider_trades,
            Modality.FILINGS_TEXT: self.filings_text,
            Modality.SUPPLY_CHAIN: self.supply_chain,
            Modality.MARKET: self.market,
            Modality.MACRO: self.macro,
            Modality.NEWS: self.news,
            Modality.PATENTS: self.patents,
            Modality.EARNINGS_CALLS: self.earnings_calls,
            Modality.HIRING: self.hiring,
        }[modality]

    def is_empty(self) -> bool:
        return all(df.empty for _, df in self._all_frames())

    def n_rows(self) -> dict[str, int]:
        return {name: len(df) for name, df in self._all_frames()}


def _assert_pit_safe(name: str, df: pd.DataFrame, as_of: date) -> None:
    """Raise if any row in `df` violates the PIT contract for `as_of`."""
    if df is None or df.empty:
        return
    if "availability_date" not in df.columns:
        raise ValueError(f"{name}: missing availability_date column — cannot enforce PIT")

    as_of_ts = pd.Timestamp(as_of)
    avail = pd.to_datetime(df["availability_date"], errors="coerce")
    if (avail > as_of_ts).any():
        offenders = df.loc[avail > as_of_ts, ["availability_date"]].head(3)
        raise AssertionError(
            f"{name}: PIT violation — {(avail > as_of_ts).sum()} rows have "
            f"availability_date > as_of={as_of}. First offenders:\n{offenders}"
        )

    if "restated_at" in df.columns:
        rs_raw = df["restated_at"]
        # rs may be all-NaN object column; coerce to datetime.
        rs = pd.to_datetime(rs_raw, errors="coerce")
        bad = rs.notna() & (rs <= as_of_ts)
        if bad.any():
            raise AssertionError(
                f"{name}: PIT violation — {bad.sum()} rows have "
                f"restated_at <= as_of={as_of} (superseded but still present)."
            )


_FIELD_FOR: dict[Modality, str] = {
    Modality.FINANCIALS: "financials",
    Modality.INSIDER_TRADES: "insider_trades",
    Modality.FILINGS_TEXT: "filings_text",
    Modality.SUPPLY_CHAIN: "supply_chain",
    Modality.MARKET: "market",
    Modality.MACRO: "macro",
    Modality.NEWS: "news",
    Modality.PATENTS: "patents",
    Modality.EARNINGS_CALLS: "earnings_calls",
    Modality.HIRING: "hiring",
}


def build_snapshot_from_pit(
    pit_engine,
    entity_id: str,
    as_of: date,
    modalities: Optional[Iterable[Modality]] = None,
) -> EntitySnapshot:
    """Build an EntitySnapshot by running PIT-bounded queries against `pit_engine`.

    Lives here (not in engine.py) to keep PITEngine free of the schema
    imports the snapshot needs. `pit_engine` is duck-typed as anything with
    a `query(modality, entity_ids, as_of, partition_keys)` method.

    Default: all 10 modalities. Pass a subset for cheaper queries.
    """
    if modalities is None:
        modalities = tuple(_FIELD_FOR.keys())
    frames: dict[str, pd.DataFrame] = {name: pd.DataFrame() for name in _FIELD_FOR.values()}
    for modality in modalities:
        df = pit_engine.query(
            modality,
            entity_ids=[entity_id],
            as_of=as_of,
            partition_keys=_lazy_partition_keys(modality),
        )
        frames[_FIELD_FOR[modality]] = df
    return EntitySnapshot(entity_id=entity_id, as_of=as_of, **frames)
