"""PIT-safe training datasets for the BWM JEPA predictor.

For each (entity, t, horizon) triple, the dataset yields:

    PITTrainingExample(
        input  = EntitySnapshot bounded at as_of=t
        target = EntitySnapshot bounded at as_of=t + horizon_quarters
    )

The two snapshots are independently PIT-bounded — neither has access to data
outside its as_of cutoff. This means training cannot leak future facts into
the input even if the dataset's tuple-generation code has a bug; the leak
would have to come from outside the snapshot API entirely, and the snapshot
API is the only path training code uses to read canonical data.

`PITDataset` is deliberately minimal — it's a plain iterable yielding
PITTrainingExample objects. Wrap it in `torch.utils.data.Dataset` later if
needed; the design keeps PyTorch out of the data layer so non-PyTorch
consumers (evaluation, analytics) can use the same primitive.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterator, Optional, Sequence

from data.pit.engine import PITEngine
from data.pit.snapshot import EntitySnapshot


def advance_quarters(d: date, n: int) -> date:
    """Add n quarters (≈ 91 days each) to `d`.

    Approximate, calendar-aware: lands on the same day-of-month of the
    target quarter when possible, else the last day of the month. We avoid
    importing dateutil to keep the dependency surface flat.
    """
    if n == 0:
        return d
    month = d.month + 3 * n
    year = d.year + (month - 1) // 12
    month = ((month - 1) % 12) + 1
    # Clamp day to a valid day in the target month.
    day = d.day
    for fallback in (day, 30, 29, 28):
        try:
            return date(year, month, fallback)
        except ValueError:
            continue
    # Should never reach here for valid Gregorian dates.
    return date(year, month, 1)


@dataclass(frozen=True)
class PITTrainingExample:
    """One (entity, t, horizon) training example with input/target snapshots.

    `input_snapshot.as_of == t` and `target_snapshot.as_of == t + horizon`.
    Both snapshots are independently PIT-bounded.

    Conventions:
      - `t` is the "current time" — the date as of which the input is built.
      - `horizon_quarters` is how many quarters forward we predict (1, 2, 4,
        8, 12 are the spec's standard horizons).
      - `target_snapshot` is a snapshot at t+h. The model trains to map the
        input snapshot to a target *latent* — the snapshot itself is the
        upstream of that target, used by the JEPA target encoder (which
        sees the same PIT-bounded ground truth).
    """

    entity_id: str
    t: date
    horizon_quarters: int
    input_snapshot: EntitySnapshot
    target_snapshot: EntitySnapshot

    def __post_init__(self) -> None:
        # Defensive consistency checks; the dataclass is frozen but creators
        # outside our control could still pass mis-bounded snapshots.
        if self.input_snapshot.entity_id != self.entity_id:
            raise ValueError("input snapshot entity_id mismatch")
        if self.target_snapshot.entity_id != self.entity_id:
            raise ValueError("target snapshot entity_id mismatch")
        if self.input_snapshot.as_of != self.t:
            raise ValueError(
                f"input snapshot as_of {self.input_snapshot.as_of} != t {self.t}"
            )
        expected_target_t = advance_quarters(self.t, self.horizon_quarters)
        if self.target_snapshot.as_of != expected_target_t:
            raise ValueError(
                f"target snapshot as_of {self.target_snapshot.as_of} != "
                f"t + {self.horizon_quarters}q = {expected_target_t}"
            )


def make_training_example(
    pit: PITEngine, entity_id: str, t: date, horizon_quarters: int
) -> PITTrainingExample:
    """Build a single PITTrainingExample via two snapshot calls.

    Each snapshot enforces its own PIT contract at construction; if either
    has any leaked rows, this function raises before returning.
    """
    target_t = advance_quarters(t, horizon_quarters)
    return PITTrainingExample(
        entity_id=entity_id,
        t=t,
        horizon_quarters=horizon_quarters,
        input_snapshot=pit.snapshot(entity_id, t),
        target_snapshot=pit.snapshot(entity_id, target_t),
    )


class PITDataset:
    """Iterable of PIT-safe training examples.

    Construction:
        pit = PITEngine(storage)
        ds = PITDataset(
            pit=pit,
            entity_ids=["us-cik-0000320193", ...],
            times=[date(2018,3,31), date(2018,6,30), ...],
            horizons=[1, 2, 4, 8, 12],
        )
        for example in ds:
            ...

    For each (entity, t) the dataset yields one example per horizon. Order
    is deterministic unless `shuffle=True` is set; with shuffle, a Random
    instance seeded from `seed` is used so runs are reproducible.

    Empty snapshots (entity with no data as-of t) are SKIPPED rather than
    raising, so the caller can pass a large universe of (entity, t) pairs
    without pre-filtering.
    """

    def __init__(
        self,
        pit: PITEngine,
        entity_ids: Sequence[str],
        times: Sequence[date],
        horizons: Sequence[int],
        *,
        shuffle: bool = False,
        seed: Optional[int] = None,
        skip_empty_input: bool = True,
    ) -> None:
        if not horizons:
            raise ValueError("horizons must be non-empty")
        self.pit = pit
        self.entity_ids = list(entity_ids)
        self.times = list(times)
        self.horizons = list(horizons)
        self.shuffle = shuffle
        self.seed = seed
        self.skip_empty_input = skip_empty_input

    def __len__(self) -> int:
        # Upper bound; actual yield count may be lower if skip_empty_input is True.
        return len(self.entity_ids) * len(self.times) * len(self.horizons)

    def __iter__(self) -> Iterator[PITTrainingExample]:
        triples = [
            (eid, t, h)
            for eid in self.entity_ids
            for t in self.times
            for h in self.horizons
        ]
        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(triples)

        for eid, t, h in triples:
            try:
                example = make_training_example(self.pit, eid, t, h)
            except AssertionError as e:
                # A PIT violation in the snapshot construction. Surface it
                # rather than swallow — this would be a real data bug.
                raise RuntimeError(
                    f"PIT violation building example ({eid}, {t}, h={h}): {e}"
                ) from e
            if self.skip_empty_input and example.input_snapshot.is_empty():
                continue
            yield example


def sample_training_times(
    start: date, end: date, frequency: str = "quarterly"
) -> list[date]:
    """Helper for constructing the `times` argument to PITDataset.

    `frequency` ∈ {"quarterly", "annual", "monthly"}. Quarterly emits the
    last day of each calendar quarter in [start, end].
    """
    out: list[date] = []
    if frequency == "quarterly":
        for q_year in range(start.year, end.year + 1):
            for q_end_month, q_end_day in ((3, 31), (6, 30), (9, 30), (12, 31)):
                d = date(q_year, q_end_month, q_end_day)
                if start <= d <= end:
                    out.append(d)
    elif frequency == "annual":
        for y in range(start.year, end.year + 1):
            d = date(y, 12, 31)
            if start <= d <= end:
                out.append(d)
    elif frequency == "monthly":
        d = date(start.year, start.month, 1)
        while d <= end:
            # Last day of the month
            next_month = (d.month % 12) + 1
            next_year = d.year + (1 if next_month == 1 else 0)
            last = date(next_year, next_month, 1) - timedelta(days=1)
            if start <= last <= end:
                out.append(last)
            d = date(next_year, next_month, 1)
    else:
        raise ValueError(f"unknown frequency: {frequency}")
    return out
