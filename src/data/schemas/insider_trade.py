"""Insider trade (SEC Form 4 / Form 4/A): one transaction by an insider.

Form 4 reports trades by officers, directors, and >10% beneficial owners,
typically filed within 2 business days of the transaction. The PIT semantics
are clean: effective_date is the transaction date, availability_date is the
filing date, so the difference is usually 0-3 days.

A single Form 4 filing carries one or more transactions; we explode them into
one row per transaction (per security) so each becomes an independent PIT
record. The (entity, insider_cik, transaction_date, security, transaction_code)
tuple uniquely identifies a transaction; amendments (Form 4/A) supersede prior
filings on the same identity.
"""
from __future__ import annotations

from datetime import date
from typing import ClassVar, Optional

from pydantic import Field

from data.schemas.pit import Modality, PITRecord


class InsiderTrade(PITRecord):
    """One reported insider transaction in the issuer's securities.

    Amendment handling: Form 4/A filings explicitly carry a `<periodOfReport>`
    that matches the original Form 4 they supersede. The ingest driver does a
    supersession sweep when writing a 4/A — it marks all canonical rows with
    the same (insider_cik, period_of_report) and an earlier accession as
    `restated_at = amendment.filed`, so as-of queries after the amendment
    only see the amendment's transactions even when transaction counts or
    indices differ between the original and the amendment.
    """

    PARTITION_KEYS: ClassVar[list[str]] = [
        "insider_cik",
        "security_title",
        "transaction_code",
        "is_derivative",
        "exercise_price",
        # Within-filing ordinal disambiguates multiple same-day same-code
        # transactions in one Form 4 (e.g. RSU vest + sell-to-cover bundled).
        # A Form 4 amendment that supersedes the original keeps the same
        # ordinal so the partition key matches and the newer row wins.
        "transaction_index",
    ]

    DEFAULT_PERSPECTIVE: ClassVar[str] = "sec_edgar:form4"
    DEFAULT_SOURCE_CERTAINTY: ClassVar[float] = 1.0

    modality: Modality = Modality.INSIDER_TRADES

    # Insider identity
    insider_cik: str = Field(..., description="CIK of the reporting owner")
    insider_name: str = Field(..., description="Reporting owner name as filed")
    is_director: bool = False
    is_officer: bool = False
    is_ten_percent_owner: bool = False
    officer_title: str = Field(default="", description="e.g. 'Chief Executive Officer'")

    # Transaction details
    security_title: str = Field(..., description="e.g. 'Common Stock', 'Stock Option'")
    is_derivative: bool = Field(
        default=False, description="True for derivative-table rows (options, RSUs)"
    )
    transaction_code: str = Field(
        ..., description="One-letter code: P/S/A/M/F/D/G/J/L/V/W/X/Z"
    )
    shares: float = Field(..., description="Number of shares (or underlying)")
    price_per_share: Optional[float] = Field(
        default=None, description="Reported price; None for grants and gifts"
    )
    acquired_disposed: str = Field(..., description="'A' (acquired) or 'D' (disposed)")
    post_transaction_shares: Optional[float] = Field(
        default=None,
        description="Shares owned following transaction (cumulative position)",
    )
    direct_or_indirect: str = Field(default="D", description="'D' direct or 'I' indirect")

    # Optional derivative-only fields
    exercise_price: Optional[float] = Field(
        default=None, description="For options/derivatives"
    )
    expiration_date: Optional[date] = Field(default=None)

    # Within-filing ordinal (assigned by the parser in document order)
    transaction_index: int = Field(
        default=0,
        description="0-based index of this transaction within the source Form 4",
    )

    # The period the report covers — Form 4's `<periodOfReport>`. Form 4/A
    # amendments carry the same period_of_report as the original they supersede;
    # used by the supersession sweep to mark old rows as restated.
    period_of_report: Optional[date] = Field(
        default=None, description="<periodOfReport> from the Form 4 XML"
    )

    # Provenance / linking
    form: str = Field(..., description="'4' or '4/A'")
    accession: str = Field(..., description="SEC accession of the Form 4 filing")
