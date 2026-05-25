"""Stage M2 driver: ingest one FRED series with full ALFRED vintage history.

Run:
    FRED_API_KEY=... uv run python -m data.cli.ingest_macro --series CPIAUCSL

Series are treated as pseudo-entities with entity_id = "fred:{series_id}".
The connector also stamps `restated_at` on superseded vintages so PIT
queries pick the value that was knowable at any given as_of date.
"""
from __future__ import annotations

import argparse
import os
from datetime import date as date_cls

import pandas as pd

from data.observability.run_id import tag as _tag
from data.pit.engine import PITEngine
from data.schemas.macro_observation import MacroObservation
from data.schemas.pit import Modality
from data.sources.edgar.archive import log_parse
from data.sources.macro.fred import DEFAULT_SERIES, fetch_series_info, fetch_vintages
from data.storage import get_storage


def _mark_restatements(rows: pd.DataFrame) -> pd.DataFrame:
    """Within (series_id, effective_date) groups, the next-later release
    supersedes the prior one. Set restated_at on each non-latest vintage.

    (effective_date is the observation date in PITRecord terms — when the
    value "was true"; availability_date is the release date when FRED
    published the value.)
    """
    if rows.empty:
        return rows
    rows = rows.sort_values(["series_id", "effective_date", "availability_date"])
    rows["restated_at"] = (
        rows.groupby(["series_id", "effective_date"])["availability_date"].shift(-1)
    )
    return rows


def ingest_series(series_id: str, *, max_observations: int = 0) -> int:
    storage = get_storage()
    pit = PITEngine(storage)
    entity_id = f"fred:{series_id}"
    info = fetch_series_info(series_id)
    print(_tag(f"  series: {series_id} — {info['title']} ({info['units']}, {info['frequency']})"))

    vintages = list(fetch_vintages(series_id))
    if not vintages:
        log_parse(storage, f"fred-{series_id}", series_id, "macro", "ok_no_records")
        return 0
    if max_observations > 0:
        vintages = vintages[:max_observations]

    rows = [
        {
            "entity_id": entity_id,
            "modality": Modality.MACRO.value,
            "effective_date": v.observation_date,
            "availability_date": v.release_date,
            "restated_at": None,
            "source": "fred:alfred",
            "source_ref": f"{v.series_id}@{v.release_date.isoformat()}",
            "series_id": v.series_id,
            "value": v.value,
            "series_title": info["title"],
            "units": info["units"],
            "frequency": info["frequency"],
        }
        for v in vintages
    ]
    df = pd.DataFrame(rows)
    df = _mark_restatements(df)
    pit.write(Modality.MACRO, df, partition_keys=MacroObservation.PARTITION_KEYS)
    log_parse(storage, f"fred-{series_id}", series_id, "macro",
              "ok_with_records", n_records=len(rows))
    print(_tag(f"  wrote {len(rows):,} vintaged observations"))
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", nargs="*", default=[],
                    help="FRED series IDs (empty → use DEFAULT_SERIES)")
    ap.add_argument("--max-observations", type=int, default=0,
                    help="cap rows per series (0 = no cap)")
    args = ap.parse_args()

    os.environ.setdefault("BWM_DATA_ROOT", ".data")
    series_list = args.series or list(DEFAULT_SERIES)
    total = 0
    for s in series_list:
        try:
            total += ingest_series(s, max_observations=args.max_observations)
        except Exception as e:  # noqa: BLE001
            print(_tag(f"  ! {s}: {type(e).__name__}: {e}"))
    print(f"\ntotal rows written: {total:,}")


if __name__ == "__main__":
    main()
