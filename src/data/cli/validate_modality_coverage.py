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

from data.storage import get_storage
from data.validation.coverage import run_coverage_checks


def main() -> int:
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

    if report.hard_failures:
        print()
        print("=== HARD FAILURES ===")
        for r in report.hard_failures:
            print(f"  - {r.name}: {r.message}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
