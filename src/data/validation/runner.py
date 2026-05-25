"""ConstraintRunner — applies bwm.accounting + bwm.regulation rules to canonical.

For each entity × (fiscal_year, fiscal_period, dimensions_json) group of
financial facts, pivot concepts to columns and evaluate every rule whose
required concepts are present. Tally hard-rule violations and produce a
CVR report against the v3 § 9.1 2% threshold.

Filing-level rules (bwm.regulation) operate on per-(entity, accession)
metadata instead of per-period concept values. The runner exposes a
`_filing_lag_days_10k` synthetic value for these rules.
"""
from __future__ import annotations

import json
from typing import Iterable, Optional

import pandas as pd

from data.pit.engine import PITEngine, _ENTITY_FILE_RE
from data.schemas.financial import FinancialFact
from data.schemas.pit import Modality
from data.storage import Storage
from data.validation.accounting import RULES as ACCOUNTING_RULES
from data.validation.regulation import RULES as REGULATION_RULES
from data.validation.rules import AccountingRule, ConstraintReport, RuleResult


class ConstraintRunner:
    def __init__(
        self,
        storage: Storage,
        pit: PITEngine,
        *,
        cvr_threshold: float = 0.02,
    ) -> None:
        self.storage = storage
        self.pit = pit
        self.cvr_threshold = cvr_threshold

    # ----- public API -----------------------------------------------------

    def evaluate(
        self,
        entity_ids: Optional[Iterable[str]] = None,
    ) -> ConstraintReport:
        """Run all rules against canonical/financials for the given entities
        (default: all). Returns a ConstraintReport."""
        report = ConstraintReport(cvr_threshold=self.cvr_threshold)
        # Initialize results for every known rule so reports show a row
        # even when a rule never matched (helps detect "rule has wrong
        # concept names" silently).
        for rule in ACCOUNTING_RULES + REGULATION_RULES:
            report.per_rule[rule.name] = RuleResult(
                rule_name=rule.name, module=rule.module, severity=rule.severity,
                modality=rule.modality,
            )

        entity_files = self._enumerate_entity_files(entity_ids)
        for path in entity_files:
            df = self.storage.read_parquet(path)
            if df.empty:
                continue
            self._evaluate_entity(df, report)
        return report

    # ----- internals ------------------------------------------------------

    def _enumerate_entity_files(
        self, entity_ids: Optional[Iterable[str]]
    ) -> list[str]:
        if entity_ids is not None:
            from data.pit.engine import _entity_path
            return [
                _entity_path(Modality.FINANCIALS, eid)
                for eid in entity_ids
                if self.storage.exists(_entity_path(Modality.FINANCIALS, eid))
            ]
        dir_path = f"canonical/{Modality.FINANCIALS.value}"
        return [
            p for p in self.storage.list(dir_path)
            if _ENTITY_FILE_RE.match(p.rsplit("/", 1)[-1])
        ]

    def _evaluate_entity(self, df: pd.DataFrame, report: ConstraintReport) -> None:
        # Exclude synthetic test fixtures from data-quality scoring — they have
        # known-impossible dates and identities by construction, and aren't
        # part of the canonical ingest the CVR gate is meant to assess.
        if "source" in df.columns:
            df = df[~df["source"].astype(str).str.startswith("test")]
            if df.empty:
                return

        # Apply restatement-aware as-of dedup: the PIT engine already does
        # this on query, but the validator wants the most-recent-known view.
        # Cheaper than going through PITEngine.query: filter to non-restated
        # and keep the latest availability_date per partition key.
        df = self._latest_view(df)
        if df.empty:
            return

        # --- Accounting rules: per-period pivot ---
        self._apply_accounting_rules(df, report)

        # --- Regulation rules: per-filing pivot ---
        self._apply_regulation_rules(df, report)

    def _latest_view(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the current as-of view of each fact: latest availability,
        no superseded rows."""
        avail = pd.to_datetime(df["availability_date"], errors="coerce")
        rs = pd.to_datetime(df.get("restated_at"), errors="coerce")
        # Drop rows that have been superseded by an earlier-released successor.
        df = df[rs.isna()]
        # Per (entity_id, effective_date) + financial partition keys, keep last.
        keys = ["entity_id", "effective_date"] + [
            k for k in FinancialFact.PARTITION_KEYS if k in df.columns
        ]
        return df.sort_values("availability_date").drop_duplicates(
            subset=keys, keep="last"
        )

    def _apply_accounting_rules(
        self, df: pd.DataFrame, report: ConstraintReport
    ) -> None:
        # Group by (entity_id, fiscal_year, fiscal_period, dimensions_json) —
        # one logical filing-period view per group. Pivot concepts within
        # the group.
        if "fiscal_year" not in df.columns:
            return
        dim_col = "dimensions_json" if "dimensions_json" in df.columns else None
        group_keys = ["entity_id", "fiscal_year", "fiscal_period"]
        if dim_col is not None:
            group_keys.append(dim_col)

        for keys, group in df.groupby(group_keys, dropna=False):
            # Pivot concept → value (take last value in case of intra-group dupes).
            values = dict(zip(group["concept"], group["value"]))
            for rule in ACCOUNTING_RULES:
                # Modality / extra_filters guard
                ef = rule.extra_filters
                if "dimensions_json" in ef and dim_col is not None:
                    dj = keys[group_keys.index(dim_col)] if isinstance(keys, tuple) else ""
                    if (dj or "") != ef["dimensions_json"]:
                        continue
                if not all(c in values for c in rule.concepts):
                    continue
                concept_values = {c: float(values[c]) for c in rule.concepts}
                result = report.per_rule[rule.name]
                result.checks += 1
                try:
                    ok = rule.check(concept_values)
                except (KeyError, ValueError, ZeroDivisionError):
                    ok = False
                if not ok:
                    result.violations += 1
                    if len(result.example_violations) < 10:
                        result.example_violations.append({
                            "entity_id": keys[0] if isinstance(keys, tuple) else keys,
                            "fiscal_year": int(keys[1]) if isinstance(keys, tuple) else None,
                            "fiscal_period": keys[2] if isinstance(keys, tuple) else None,
                            "concepts": concept_values,
                        })

    def _apply_regulation_rules(
        self, df: pd.DataFrame, report: ConstraintReport
    ) -> None:
        # Per-filing: lag must be computed from the *latest* fact's effective
        # date in the filing (the fiscal year actually being reported), not
        # the earliest — 10-Ks include comparative facts from prior years
        # whose effective_dates are 1-3 years before the filing date.
        if "accession" not in df.columns or "form" not in df.columns:
            return
        avail = pd.to_datetime(df["availability_date"], errors="coerce")
        eff = pd.to_datetime(df["effective_date"], errors="coerce")
        per_filing = (
            df.assign(_avail=avail, _eff=eff)
            .groupby(["entity_id", "accession"], dropna=False)
            .agg(
                form=("form", "first"),
                report_date=("_eff", "max"),
                filed_date=("_avail", "max"),
            )
            .reset_index()
        )
        per_filing["filing_lag_days"] = (
            per_filing["filed_date"] - per_filing["report_date"]
        ).dt.days

        for rule in REGULATION_RULES:
            ef = rule.extra_filters or {}
            form_filter = ef.get("_form_is")
            scoped = per_filing
            if form_filter is not None:
                scoped = scoped[scoped["form"] == form_filter]
            for row in scoped.itertuples(index=False):
                if pd.isna(row.filing_lag_days):
                    continue
                values = {"_filing_lag_days_10k": int(row.filing_lag_days)}
                result = report.per_rule[rule.name]
                result.checks += 1
                try:
                    ok = rule.check(values)
                except (KeyError, ValueError, ZeroDivisionError):
                    ok = False
                if not ok:
                    result.violations += 1
                    if len(result.example_violations) < 10:
                        result.example_violations.append({
                            "entity_id": row.entity_id,
                            "accession": row.accession,
                            "form": row.form,
                            "report_date": str(row.report_date.date() if hasattr(row.report_date, 'date') else row.report_date),
                            "filed_date": str(row.filed_date.date() if hasattr(row.filed_date, 'date') else row.filed_date),
                            "lag_days": int(row.filing_lag_days),
                        })
