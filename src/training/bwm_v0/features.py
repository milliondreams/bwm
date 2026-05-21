"""EntitySnapshot → tensor: pivot consolidated financials to (T, n_concepts).

Pure function over an `EntitySnapshot.financials` DataFrame. No engine
queries. PIT correctness is inherited from the snapshot's own as_of bound —
this module only reshapes, never queries. That separation is important: it
means PIT integrity holds *by construction* in the training loop, since
every tensor was made from a bounded snapshot.
"""
from __future__ import annotations

import numpy as np

from data.pit.snapshot import EntitySnapshot
from training.bwm_v0.concepts import CONCEPT_TO_IDX, N_CONCEPTS, normalize_value


# Quarter ordering within a fiscal year. FY (annual) appears after the four
# quarters — when both Q4 and FY values exist for the same year, the later
# FY value (from the 10-K) takes precedence over the implied Q4 (from the
# 10-Q). The "Q" fallback covers per-filing iXBRL that hasn't been classified.
_PERIOD_ORDER: dict[str, int] = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4, "FY": 5, "Q": 1}


def _period_sort_key(fiscal_year, fiscal_period) -> tuple[int, int]:
    """Sort key so the temporal order in the tensor is correct."""
    return (int(fiscal_year), _PERIOD_ORDER.get(str(fiscal_period), 0))


def snapshot_to_tensor(
    snapshot: EntitySnapshot,
    max_history_q: int = 32,
) -> dict[str, np.ndarray | int]:
    """Convert a snapshot's financials into a fixed-shape feature tensor.

    Returns:
        x:       float32 (max_history_q, N_CONCEPTS), normalized values
        mask:    int64   (max_history_q,)             1 for real periods, 0 for padding
        seq_len: int                                   number of real periods (≤ max_history_q)

    Padding: left-pad if shorter (real data at the END of the tensor; the
    model sees most-recent quarters at high positions). Truncate from the
    start if longer (drop oldest history).

    Filter: consolidated rows only (dimensions_json == ""). Segment-level
    rows would confuse the temporal pivot since they describe different
    accounting scope at the same period.
    """
    empty_result: dict[str, np.ndarray | int] = {
        "x": np.zeros((max_history_q, N_CONCEPTS), dtype=np.float32),
        "mask": np.zeros((max_history_q,), dtype=np.int64),
        "seq_len": 0,
    }
    fin = snapshot.financials
    if fin is None or fin.empty:
        return empty_result

    # Filter consolidated + curated concepts.
    if "dimensions_json" in fin.columns:
        fin = fin[fin["dimensions_json"].fillna("") == ""]
    fin = fin[fin["concept"].isin(CONCEPT_TO_IDX.keys())]
    if fin.empty:
        return empty_result

    fin = fin.copy()
    # Tuple keys: (fiscal_year, fiscal_period). Use the tuple as the period id.
    fin["_period_key"] = list(zip(
        fin["fiscal_year"].astype(int), fin["fiscal_period"].astype(str)
    ))
    fin["_period_order"] = [
        _period_sort_key(fy, fp)
        for fy, fp in zip(fin["fiscal_year"], fin["fiscal_period"])
    ]

    # Sort by availability_date so drop_duplicates(keep='last') keeps the most
    # recently filed value per (period, concept). This is the as-of view at
    # the snapshot's as_of date (the snapshot was already PIT-bounded upstream).
    fin = fin.sort_values("availability_date")
    fin = fin.drop_duplicates(subset=["_period_key", "concept"], keep="last")

    # Now sort temporally for output positioning.
    fin = fin.sort_values("_period_order")

    # Unique periods in temporal order.
    seen = set()
    periods: list = []
    for pk in fin["_period_key"]:
        if pk in seen:
            continue
        seen.add(pk)
        periods.append(pk)

    # Truncate from the start if too long (drop oldest).
    if len(periods) > max_history_q:
        periods = periods[-max_history_q:]
        fin = fin[fin["_period_key"].isin(periods)]

    x = np.zeros((max_history_q, N_CONCEPTS), dtype=np.float32)
    mask = np.zeros((max_history_q,), dtype=np.int64)
    # Left-pad: real periods land at the END of the tensor.
    offset = max_history_q - len(periods)
    period_to_row = {pk: offset + i for i, pk in enumerate(periods)}

    for row in fin.itertuples(index=False):
        pk = (int(row.fiscal_year), str(row.fiscal_period))
        r = period_to_row.get(pk)
        if r is None:
            continue
        c = CONCEPT_TO_IDX.get(row.concept)
        if c is None:
            continue
        x[r, c] = normalize_value(float(row.value))

    for pk in periods:
        mask[period_to_row[pk]] = 1

    return {"x": x, "mask": mask, "seq_len": len(periods)}
