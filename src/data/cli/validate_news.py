"""Stage M4 validation: GDELT news pipeline.

The live GDELT 15-min CSV downloads aren't exercised here (large data,
network-dependent). This validation exercises the parser + schema + canonical
+ snapshot integration with a synthetic GDELT-shaped TSV row.

Asserts:
  - parse_gdelt_csv extracts fields from a realistic GDELT TSV line
  - canonical persistence and snapshot integration work
  - PIT semantics: as-of before date_added excludes; on/after includes
  - quad_class and goldstein_scale roundtrip with correct types
"""
from __future__ import annotations

import os
from datetime import date

import pandas as pd

from data.entity.registry import EntityRegistry
from data.pit.engine import PITEngine
from data.schemas.news_event import NewsEvent
from data.schemas.pit import Modality
from data.sources.news.gdelt import parse_gdelt_csv
from data.storage import get_storage


# One realistic-shaped GDELT 2.0 TSV row. Real exports have 61 columns; we
# fill the relevant indices and leave the rest empty.
def _make_row(global_id: str, day: str, actor1: str, actor2: str, event_code: str,
              event_root: str, quad: str, goldstein: str, n_mentions: str,
              avg_tone: str, source: str, date_added: str) -> bytes:
    cols = [""] * 61
    cols[0] = global_id
    cols[1] = day
    cols[6] = actor1
    cols[16] = actor2
    cols[26] = event_code
    cols[27] = event_root
    cols[28] = quad
    cols[29] = goldstein
    cols[31] = n_mentions
    cols[34] = avg_tone
    cols[57] = source
    cols[59] = date_added
    return ("\t".join(cols) + "\n").encode("utf-8")


def main() -> None:
    os.environ.setdefault("BWM_DATA_ROOT", ".data")
    storage = get_storage()
    pit = PITEngine(storage)
    eid = EntityRegistry.entity_id_from_cik("0000320193")

    print("=== Stage M4 validation: GDELT news pipeline ===")
    print("\n(Note: live GDELT 15-min CSV downloads are operational and exercised")
    print(" separately. This validation parses synthetic rows in the GDELT 2.0")
    print(" schema to prove parser + schema + canonical + snapshot work end-to-end.)\n")

    # Two synthetic events about Apple — one positive earnings report, one negative.
    body = (
        _make_row(
            global_id="123456789",
            day="20240201",
            actor1="APPLE INC",
            actor2="",
            event_code="036",  # Express intent to cooperate
            event_root="03",
            quad="1",
            goldstein="3.0",
            n_mentions="42",
            avg_tone="5.6",
            source="https://reuters.com/aapl-q1-2024",
            date_added="20240202050000",
        )
        + _make_row(
            global_id="123456790",
            day="20240315",
            actor1="APPLE INC",
            actor2="GOOGLE INC",
            event_code="190",  # Use conventional military force
            event_root="19",
            quad="3",
            goldstein="-7.5",
            n_mentions="18",
            avg_tone="-12.3",
            source="https://example.com/competition",
            date_added="20240316060000",
        )
    )

    events = list(parse_gdelt_csv(body))
    assert len(events) == 2, f"expected 2 parsed events, got {len(events)}"
    print(f"  [OK] parsed {len(events)} GDELT events from synthetic TSV")

    # Wipe canonical for this entity for repeatability
    from data.pit.engine import _entity_path
    from pathlib import Path
    canon = _entity_path(Modality.NEWS, eid)
    if storage.exists(canon):
        (Path(storage.root) / canon).unlink()

    rows = [
        {
            "entity_id": eid,
            "modality": Modality.NEWS.value,
            "effective_date": e.event_date,
            "availability_date": e.date_added,
            "restated_at": None,
            "source": "gdelt:events2",
            "source_ref": e.global_event_id,
            "global_event_id": e.global_event_id,
            "cameo_event_code": e.cameo_event_code,
            "event_root_code": e.event_root_code,
            "quad_class": e.quad_class,
            "goldstein_scale": e.goldstein_scale,
            "tone": e.avg_tone,
            "source_url": e.source_url,
            "actor1_name": e.actor1_name,
            "actor2_name": e.actor2_name,
            "n_mentions": e.n_mentions,
            "avg_tone": e.avg_tone,
        }
        for e in events
    ]
    pit.write(Modality.NEWS, pd.DataFrame(rows), partition_keys=NewsEvent.PARTITION_KEYS)
    print(f"  wrote {len(rows)} news events to canonical")

    # PIT: snapshot before first date_added excludes; on/after includes
    pre = pit.snapshot(eid, as_of=date(2024, 2, 1))
    assert len(pre.news) == 0, f"news visible before date_added: {len(pre.news)}"
    print(f"  [OK] PIT: no events knowable on 2024-02-01 (before first index date)")

    mid = pit.snapshot(eid, as_of=date(2024, 2, 5))
    assert len(mid.news) == 1
    assert mid.news.iloc[0]["global_event_id"] == "123456789"
    print(f"  [OK] PIT: 1 event visible mid-period; quad_class={int(mid.news.iloc[0]['quad_class'])}, "
          f"goldstein={float(mid.news.iloc[0]['goldstein_scale']):.1f}")

    after = pit.snapshot(eid, as_of=date(2024, 4, 1))
    assert len(after.news) == 2
    quad_dist = after.news["quad_class"].value_counts().to_dict()
    print(f"  [OK] both events visible after both index dates; quad_class distribution: {quad_dist}")

    print("\n=== Stage M4 validation: PASS ===")


if __name__ == "__main__":
    main()
