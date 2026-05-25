"""Phase A acceptance gate: per-modality coverage report.

Complements `validate_constraints` (correctness) with operational coverage
checks: do we have ENOUGH data to declare Phase A complete?

Exit code:
    0 if all hard checks pass
    1 if any hard check fails (soft failures are reported as WARN)

Writes a structured JSON report to `validation/modality_coverage.json` so the
AML pipeline orchestrator can gate the swap step on coverage outcomes.

Run:
    BWM_DATA_ROOT=.data uv run python -m data.cli.validate_modality_coverage
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict

from data.observability.baseline import (
    diff,
    has_material_drift,
    latest_baseline,
    write_baseline,
)
from data.observability.log import emit_event
from data.storage import get_storage
from data.validation.coverage import run_coverage_checks


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-diff-baseline", action="store_true",
                    help="skip comparing against prior baseline and skip promotion")
    args = ap.parse_args()

    storage = get_storage()
    report = run_coverage_checks(storage)

    print(f"{'check':<48} {'sev':<6} {'pass':<6} observed")
    print("-" * 100)
    for r in report.results:
        status = "PASS" if r.passed else ("FAIL" if r.severity == "hard" else "WARN")
        print(f"{r.name:<48} {r.severity:<6} {status:<6} {r.observed}")

    print()
    print(f"Hard failures: {len(report.hard_failures)}")
    print(f"Soft failures: {len(report.soft_failures)}")
    print(f"Gate:          {'PASS' if report.passes else 'FAIL'}")

    # Persist the structured report for AML pipeline gating
    json_report = {
        "passes": report.passes,
        "results": [asdict(r) for r in report.results],
    }
    try:
        storage.write_bytes(
            "validation/modality_coverage.json",
            json.dumps(json_report, indent=2).encode("utf-8"),
        )
    except Exception:
        # Validation reporting should never block the CLI exit code
        pass

    # Emit per-check events (machine-consumable) + aggregate
    for r in report.results:
        emit_event(
            "validate_modality_coverage", "all",
            "check_pass" if r.passed else "check_fail",
            check_name=r.name, severity=r.severity, observed=r.observed,
        )
    emit_event(
        "validate_modality_coverage", "all",
        "validation_pass" if report.passes else "validation_fail",
        hard_failures=len(report.hard_failures),
        soft_failures=len(report.soft_failures),
        n_checks=len(report.results),
    )

    # Baseline comparison + auto-promotion on pass. Coverage baselines live
    # under the same state/baselines/ directory keyed by ISO timestamp; they
    # use a `coverage` field that diff() consumes generically.
    if not args.no_diff_baseline:
        current_payload = {
            "coverage": {
                r.name: {
                    "observed": r.observed, "passed": r.passed,
                    "severity": r.severity,
                }
                for r in report.results
            },
        }
        prior = latest_baseline(storage)
        if prior is not None:
            prior_path, prior_payload = prior
            d = diff(prior_payload, current_payload)
            changed = [k for k, v in d["coverage"].items() if v["changed"]]
            emit_event(
                "validate_modality_coverage", "all", "regression_diff",
                prior_baseline=prior_path, changed_checks=changed,
                material_drift=has_material_drift(d),
            )
            if changed:
                print("\n=== Δ vs prior baseline ===")
                for k in changed:
                    v = d["coverage"][k]
                    print(f"  {k}: {v['prior_observed']} → {v['current_observed']}")
        if report.passes:
            path = write_baseline(storage, current_payload)
            emit_event(
                "validate_modality_coverage", "all", "baseline_promoted",
                baseline_path=path,
            )

    if report.hard_failures:
        print()
        print("=== HARD FAILURES ===")
        for r in report.hard_failures:
            print(f"  - {r.name}: {r.message}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
