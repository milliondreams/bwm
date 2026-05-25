"""Baseline + regression-diff infrastructure for validation runs.

Every passing validation run writes a JSON baseline under
`state/baselines/{YYYYMMDDTHHMMSSZ}.json`. The next run compares its
current payload against the most-recent baseline and emits a
`regression_diff` event with absolute + relative deltas on the metrics
that matter (CVR, hard checks, per-modality checks/violations, coverage
row counts).

Auto-promotion: failing runs do NOT overwrite the prior good baseline,
so a regression that trips the gate can't silently become the new
normal. Passing runs always promote.

Single shape across `validate_constraints` and `validate_modality_coverage`
so the diff function is modality-agnostic.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from data.storage import Storage

BASELINE_DIR = "state/baselines"
_BASELINE_NAME_RE = re.compile(r"^(\d{8}T\d{6}Z)\.json$")


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _baseline_path(ts: str) -> str:
    return f"{BASELINE_DIR}/{ts}.json"


def write_baseline(storage: Storage, payload: dict) -> str:
    """Serialize `payload` (plus a current UTC ts) under state/baselines/.

    Returns the storage-relative path of the new file. Caller decides
    whether to call this (i.e., only on passing runs).
    """
    ts = _utc_now_compact()
    body = dict(payload)
    body.setdefault("ts", datetime.now(timezone.utc).isoformat())
    path = _baseline_path(ts)
    storage.write_bytes(path, json.dumps(body, indent=2, default=str).encode("utf-8"))
    return path


def latest_baseline(storage: Storage) -> tuple[str, dict] | None:
    """Return (path, payload) of the lexicographically-greatest baseline file.

    Filenames are ISO8601 compact UTC, so lex-sort equals chronological sort.
    Returns None if the directory is empty or missing.
    """
    try:
        entries = list(storage.list(BASELINE_DIR))
    except Exception:
        return None
    candidates: list[str] = []
    for p in entries:
        name = p.rsplit("/", 1)[-1]
        if _BASELINE_NAME_RE.match(name):
            candidates.append(p)
    if not candidates:
        return None
    latest = max(candidates)
    try:
        payload = json.loads(storage.read_bytes(latest).decode("utf-8"))
    except Exception:
        return None
    return latest, payload


def _relative_delta(prior: float, current: float) -> float:
    """(current - prior) / max(|prior|, 1) — tolerates zero-prior cleanly."""
    denom = abs(prior) if abs(prior) > 1.0 else 1.0
    return (current - prior) / denom


def diff(prior: dict, current: dict) -> dict:
    """Compare two baseline payloads; return a structured diff.

    The diff has three sections:
      - scalars: top-level numerics (cvr, hard_checks, hard_violations)
      - per_modality: dict[modality, dict[metric, {prior, current, delta, rel_delta}]]
      - coverage: dict[check_name, {prior_observed, current_observed, changed: bool}]

    The diff is intentionally numeric — operators decide thresholds.
    """
    out: dict = {"scalars": {}, "per_modality": {}, "coverage": {}}

    for key in ("cvr", "hard_checks", "hard_violations"):
        p = float(prior.get(key, 0) or 0)
        c = float(current.get(key, 0) or 0)
        out["scalars"][key] = {
            "prior": p, "current": c,
            "delta": c - p, "rel_delta": _relative_delta(p, c),
        }

    prior_pm = prior.get("per_modality") or {}
    current_pm = current.get("per_modality") or {}
    all_mods = set(prior_pm) | set(current_pm)
    for m in sorted(all_mods):
        pm = prior_pm.get(m) or {}
        cm = current_pm.get(m) or {}
        out["per_modality"][m] = {}
        for metric in ("checks", "violations", "cvr"):
            p = float(pm.get(metric, 0) or 0)
            c = float(cm.get(metric, 0) or 0)
            out["per_modality"][m][metric] = {
                "prior": p, "current": c,
                "delta": c - p, "rel_delta": _relative_delta(p, c),
            }

    prior_cov = prior.get("coverage") or {}
    current_cov = current.get("coverage") or {}
    all_checks = set(prior_cov) | set(current_cov)
    for name in sorted(all_checks):
        po = prior_cov.get(name) or {}
        co = current_cov.get(name) or {}
        out["coverage"][name] = {
            "prior_observed": po.get("observed"),
            "current_observed": co.get("observed"),
            "changed": po.get("observed") != co.get("observed"),
        }
    return out


def has_material_drift(d: dict, threshold: float = 0.05) -> bool:
    """True if any scalar or per-modality cvr/checks rel_delta exceeds threshold."""
    for entry in d["scalars"].values():
        if abs(entry["rel_delta"]) > threshold:
            return True
    for mod in d["per_modality"].values():
        for entry in mod.values():
            if abs(entry["rel_delta"]) > threshold:
                return True
    return False
