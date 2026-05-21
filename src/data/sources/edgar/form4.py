"""SEC Form 4 (insider transactions) XML parser.

A Form 4 filing has a primary "ownership document" — an XML file (recent) or
a TXT block (older filings) — with the structure:

    <ownershipDocument>
      <issuer><issuerCik>...</issuerCik></issuer>
      <reportingOwner>
        <reportingOwnerId><rptOwnerCik>...</rptOwnerCik><rptOwnerName>...</rptOwnerName></reportingOwnerId>
        <reportingOwnerRelationship>
          <isDirector>1</isDirector><isOfficer>0</isOfficer>...
          <officerTitle>CEO</officerTitle>
        </reportingOwnerRelationship>
      </reportingOwner>
      <nonDerivativeTable>
        <nonDerivativeTransaction>
          <securityTitle><value>Common Stock</value></securityTitle>
          <transactionDate><value>2023-10-01</value></transactionDate>
          <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
          <transactionAmounts>
            <transactionShares><value>500</value></transactionShares>
            <transactionPricePerShare><value>170.00</value></transactionPricePerShare>
            <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
          </transactionAmounts>
          <postTransactionAmounts>
            <sharesOwnedFollowingTransaction><value>3287045</value></sharesOwnedFollowingTransaction>
          </postTransactionAmounts>
          <ownershipNature>
            <directOrIndirectOwnership><value>D</value></directOrIndirectOwnership>
          </ownershipNature>
        </nonDerivativeTransaction>
      </nonDerivativeTable>
      <derivativeTable>...</derivativeTable>
    </ownershipDocument>

A single filing can have many transactions (especially equity grants vest +
sell-to-cover happening together). We yield one ParsedTrade per transaction
in both nonDerivative and derivative tables.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterator, Optional

from lxml import etree


@dataclass(frozen=True)
class ParsedForm4:
    """The full parsed view of one Form 4 filing.

    `period_of_report` enables the ingest driver's supersession sweep on
    amendments: a Form 4/A and the original Form 4 it amends share the same
    period_of_report.
    """
    period_of_report: Optional[date]
    trades: list  # list[ParsedTrade] — circular type avoided

    def __iter__(self):
        return iter(self.trades)


@dataclass(frozen=True)
class ParsedTrade:
    insider_cik: str
    insider_name: str
    is_director: bool
    is_officer: bool
    is_ten_percent_owner: bool
    officer_title: str

    security_title: str
    is_derivative: bool
    transaction_date: date
    transaction_code: str
    shares: float
    price_per_share: Optional[float]
    acquired_disposed: str
    post_transaction_shares: Optional[float]
    direct_or_indirect: str
    exercise_price: Optional[float]
    expiration_date: Optional[date]


def _text(el: Optional[etree._Element]) -> str:
    if el is None:
        return ""
    # Most fields wrap their value in a <value>...</value> element.
    val = el.find("value")
    if val is not None and val.text is not None:
        return val.text.strip()
    return (el.text or "").strip()


def _bool(el: Optional[etree._Element]) -> bool:
    s = _text(el).lower()
    return s in ("1", "true")


def _date(el: Optional[etree._Element]) -> Optional[date]:
    s = _text(el)
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except ValueError:
        return None


def _float(el: Optional[etree._Element]) -> Optional[float]:
    s = _text(el)
    if not s:
        return None
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def _parse_owner(doc: etree._Element) -> dict:
    """Return a dict of insider attributes shared across transactions."""
    owner = doc.find("reportingOwner")
    if owner is None:
        return {}
    ident = owner.find("reportingOwnerId")
    rel = owner.find("reportingOwnerRelationship")
    return {
        "insider_cik": (_text(ident.find("rptOwnerCik")) if ident is not None else "").zfill(10),
        "insider_name": _text(ident.find("rptOwnerName")) if ident is not None else "",
        "is_director": _bool(rel.find("isDirector")) if rel is not None else False,
        "is_officer": _bool(rel.find("isOfficer")) if rel is not None else False,
        "is_ten_percent_owner": _bool(rel.find("isTenPercentOwner")) if rel is not None else False,
        "officer_title": _text(rel.find("officerTitle")) if rel is not None else "",
    }


def _parse_transaction(tx: etree._Element, derivative: bool, owner: dict) -> Optional[ParsedTrade]:
    sec = tx.find("securityTitle")
    if sec is None:
        return None
    tx_date_el = tx.find("transactionDate")
    if tx_date_el is None:
        return None
    tx_date = _date(tx_date_el)
    if tx_date is None:
        return None

    coding = tx.find("transactionCoding")
    code = _text(coding.find("transactionCode")) if coding is not None else ""

    amounts = tx.find("transactionAmounts")
    shares = _float(amounts.find("transactionShares")) if amounts is not None else None
    if shares is None:
        return None
    price = _float(amounts.find("transactionPricePerShare")) if amounts is not None else None
    ad = _text(amounts.find("transactionAcquiredDisposedCode")) if amounts is not None else ""

    post = tx.find("postTransactionAmounts")
    post_shares = _float(post.find("sharesOwnedFollowingTransaction")) if post is not None else None

    nature = tx.find("ownershipNature")
    doi = _text(nature.find("directOrIndirectOwnership")) if nature is not None else "D"

    exercise_price: Optional[float] = None
    expiration: Optional[date] = None
    if derivative:
        exercise_price = _float(tx.find("conversionOrExercisePrice"))
        expiration = _date(tx.find("expirationDate"))

    return ParsedTrade(
        insider_cik=owner.get("insider_cik", ""),
        insider_name=owner.get("insider_name", ""),
        is_director=owner.get("is_director", False),
        is_officer=owner.get("is_officer", False),
        is_ten_percent_owner=owner.get("is_ten_percent_owner", False),
        officer_title=owner.get("officer_title", ""),
        security_title=_text(sec),
        is_derivative=derivative,
        transaction_date=tx_date,
        transaction_code=code,
        shares=shares,
        price_per_share=price,
        acquired_disposed=ad,
        post_transaction_shares=post_shares,
        direct_or_indirect=doi,
        exercise_price=exercise_price,
        expiration_date=expiration,
    )


def parse_form4(xml_body: bytes) -> Iterator[ParsedTrade]:
    """Yield ParsedTrade rows for every transaction in a Form 4 XML doc.

    Kept for back-compat with code that only needs trades. New callers should
    prefer `parse_form4_filing` which also returns the periodOfReport needed
    for amendment supersession.
    """
    parsed = parse_form4_filing(xml_body)
    if parsed is None:
        return
    yield from parsed.trades


def parse_form4_filing(xml_body: bytes) -> Optional[ParsedForm4]:
    """Parse a Form 4 XML body into trades + filing-level metadata.

    Returns None if the document is unparseable or has no reporting owner.
    """
    try:
        doc = etree.fromstring(xml_body)
    except etree.XMLSyntaxError:
        # Some older filings have a TXT wrapper; try to locate the XML block.
        start = xml_body.find(b"<ownershipDocument")
        end = xml_body.find(b"</ownershipDocument>")
        if start < 0 or end < 0:
            return None
        try:
            doc = etree.fromstring(xml_body[start : end + len(b"</ownershipDocument>")])
        except etree.XMLSyntaxError:
            return None

    owner = _parse_owner(doc)
    if not owner.get("insider_cik"):
        return None

    period_of_report = _date(doc.find("periodOfReport"))

    trades: list[ParsedTrade] = []
    for table_tag, is_deriv in (
        ("nonDerivativeTable", False),
        ("derivativeTable", True),
    ):
        table = doc.find(table_tag)
        if table is None:
            continue
        tx_tag = "nonDerivativeTransaction" if not is_deriv else "derivativeTransaction"
        for tx in table.findall(tx_tag):
            parsed = _parse_transaction(tx, is_deriv, owner)
            if parsed is not None:
                trades.append(parsed)

    return ParsedForm4(period_of_report=period_of_report, trades=trades)
