"""CVR gate: bwm.accounting + bwm.regulation hard rules.

v3 § 9.1 Phase A acceptance: hard CVR ≤ 2%.

Run:
    uv run python -m data.cli.validate_constraints
    uv run python -m data.cli.validate_constraints --entity-id us-cik-0000320193
    uv run python -m data.cli.validate_constraints --threshold 0.05

Exits 0 if CVR ≤ threshold; 1 otherwise.
"""
from __future__ import annotations

import argparse
import os
import sys

from data.pit.engine import PITEngine
from data.storage import get_storage
from data.validation.runner import ConstraintRunner


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity-id", default="", help="scope to one entity (default: all)")
    ap.add_argument("--threshold", type=float, default=0.02,
                    help="CVR gate threshold (default: 0.02 per v3 § 9.1)")
    ap.add_argument("--show-examples", type=int, default=3,
                    help="number of example violations to print per failing rule")
    args = ap.parse_args()

    os.environ.setdefault("BWM_DATA_ROOT", ".data")
    storage = get_storage()
    pit = PITEngine(storage)
    runner = ConstraintRunner(storage, pit, cvr_threshold=args.threshold)

    print("=== CVR validation: bwm.accounting + bwm.regulation ===\n")
    entity_ids = [args.entity_id] if args.entity_id else None
    report = runner.evaluate(entity_ids=entity_ids)

    # Per-rule table
    print(f"{'rule':<48} {'sev':<5} {'module':<16} {'checks':>10} {'viols':>8} {'rate':>8}")
    print("─" * 100)
    for name, result in report.per_rule.items():
        print(
            f"{name:<48} {result.severity:<5} {result.module:<16} "
            f"{result.checks:>10,} {result.violations:>8,} {result.violation_rate:>8.3%}"
        )

    # Aggregate CVR
    print()
    print(f"Hard checks:     {report.total_hard_checks:,}")
    print(f"Hard violations: {report.total_hard_violations:,}")
    print(f"CVR:             {report.cvr:.4%}")
    print(f"Threshold:       {report.cvr_threshold:.2%}")
    print(f"Gate:            {'PASS' if report.passes_gate else 'FAIL'}")

    # Example violations for failing rules
    if args.show_examples > 0:
        any_violations = any(r.violations > 0 for r in report.per_rule.values())
        if any_violations:
            print("\n=== example violations ===")
            for name, result in report.per_rule.items():
                if result.violations == 0:
                    continue
                print(f"\n[{name}] ({result.module}, sev={result.severity}) — "
                      f"{result.violations}/{result.checks} violations")
                for ex in result.example_violations[: args.show_examples]:
                    print(f"  - {ex}")

    raise SystemExit(0 if report.passes_gate else 1)


if __name__ == "__main__":
    main()
