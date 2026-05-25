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

from data.observability.baseline import (
    diff,
    has_material_drift,
    latest_baseline,
    write_baseline,
)
from data.observability.log import emit_event
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
    ap.add_argument("--no-diff-baseline", action="store_true",
                    help="skip comparing against prior baseline and skip promotion")
    args = ap.parse_args()

    os.environ.setdefault("BWM_DATA_ROOT", ".data")
    storage = get_storage()
    pit = PITEngine(storage)
    runner = ConstraintRunner(storage, pit, cvr_threshold=args.threshold)

    print("=== CVR validation: bwm.accounting + bwm.regulation ===\n")
    entity_ids = [args.entity_id] if args.entity_id else None
    report = runner.evaluate(entity_ids=entity_ids)

    # Per-rule table
    print(f"{'rule':<48} {'sev':<5} {'module':<16} {'modality':<12} {'checks':>10} {'viols':>8} {'rate':>8}")
    print("─" * 112)
    for name, result in report.per_rule.items():
        print(
            f"{name:<48} {result.severity:<5} {result.module:<16} "
            f"{result.modality:<12} "
            f"{result.checks:>10,} {result.violations:>8,} {result.violation_rate:>8.3%}"
        )

    # Per-modality summary (hard rules only — matches gate semantics)
    print()
    print(f"{'modality':<16} {'checks':>10} {'viols':>8} {'cvr':>8}")
    print("─" * 46)
    for modality, agg in report.per_modality.items():
        print(
            f"{modality:<16} {agg['checks']:>10,} "
            f"{agg['violations']:>8,} {agg['cvr']:>8.3%}"
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

    emit_event(
        "validate_constraints", "financials",
        "validation_pass" if report.passes_gate else "validation_fail",
        cvr=report.cvr, threshold=report.cvr_threshold,
        hard_checks=report.total_hard_checks,
        hard_violations=report.total_hard_violations,
        passes_gate=report.passes_gate,
    )
    for modality, agg in report.per_modality.items():
        emit_event(
            "validate_constraints", modality, "modality_breakdown",
            cvr=agg["cvr"], checks=agg["checks"], violations=agg["violations"],
        )

    # Baseline comparison + auto-promotion on pass.
    if not args.no_diff_baseline:
        current_payload = {
            "cvr": report.cvr,
            "hard_checks": report.total_hard_checks,
            "hard_violations": report.total_hard_violations,
            "per_modality": report.per_modality,
        }
        prior = latest_baseline(storage)
        if prior is not None:
            prior_path, prior_payload = prior
            d = diff(prior_payload, current_payload)
            material = has_material_drift(d)
            emit_event(
                "validate_constraints", "financials", "regression_diff",
                prior_baseline=prior_path,
                material_drift=material,
                cvr_rel_delta=d["scalars"]["cvr"]["rel_delta"],
                hard_checks_rel_delta=d["scalars"]["hard_checks"]["rel_delta"],
            )
            if material:
                print("\n=== Δ vs prior baseline (material drift > 5%) ===")
                for k, v in d["scalars"].items():
                    if abs(v["rel_delta"]) > 0.05:
                        print(f"  {k}: {v['prior']:.4f} → {v['current']:.4f} "
                              f"(Δ {v['delta']:+.4f}, rel {v['rel_delta']:+.2%})")
        if report.passes_gate:
            path = write_baseline(storage, current_payload)
            emit_event(
                "validate_constraints", "financials", "baseline_promoted",
                baseline_path=path,
            )

    raise SystemExit(0 if report.passes_gate else 1)


if __name__ == "__main__":
    main()
