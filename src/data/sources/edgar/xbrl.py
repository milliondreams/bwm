"""Inline-XBRL (iXBRL) parser for SEC filings.

Modern SEC filings (post-~2020 for most filers) embed XBRL inline in the HTML
primary document. The XBRL contexts and units are declared in an `<ix:header>`
block (usually inside `<ix:hidden>`), and each fact is an `<ix:nonFraction>`
or `<ix:nonNumeric>` element in the document body referencing those contexts.

This module is intentionally focused: it extracts numeric financial facts with
their full XBRL context dimensions (segment / scenario / typed dimensions),
which is exactly what `companyfacts` flattens away.

What we extract:
  - concept name (e.g. "us-gaap:Revenues")
  - value as float (decimals + scale applied)
  - unit (e.g. "USD", "shares")
  - period (instant date OR start/end pair)
  - context_id (the XBRL contextRef)
  - dimensions: dict[axis_qname → member_qname]

What we deliberately skip:
  - Non-numeric facts (`ix:nonNumeric`) — text/narrative content is Stage 4
  - Typed dimensions — rare; deferred to a hardening pass
  - `xsi:nil="true"` facts — these mean "no value reported"
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterator, Optional

from lxml import etree, html as lxml_html

# lxml's HTML parser doesn't produce Clark-style {URI}localname tags for
# namespaced elements (because HTML doesn't have proper namespace handling).
# It keeps `prefix:localname` as a single string, lowercased. We therefore
# match by string equality on the lowercased qname. The SEC's iXBRL docs
# use stable prefixes (ix, xbrli, xbrldi), so this is reliable.
TAG_IX_NONFRACTION = {"ix:nonfraction"}     # iXBRL 1.1 and 1.0 use same local name
TAG_XBRLI_CONTEXT = "xbrli:context"
TAG_XBRLI_PERIOD = "xbrli:period"
TAG_XBRLI_INSTANT = "xbrli:instant"
TAG_XBRLI_START = "xbrli:startdate"
TAG_XBRLI_END = "xbrli:enddate"
TAG_XBRLI_SEGMENT = "xbrli:segment"
TAG_XBRLI_SCENARIO = "xbrli:scenario"
TAG_XBRLI_UNIT = "xbrli:unit"
TAG_XBRLI_MEASURE = "xbrli:measure"
TAG_XBRLI_DIVIDE = "xbrli:divide"
TAG_XBRLDI_EXPLICIT = "xbrldi:explicitmember"

# Attributes — lxml HTML parser also lowercases attribute names.
ATTR_NIL = "{http://www.w3.org/2001/XMLSchema-instance}nil"
ATTR_NIL_LOWER = "nil"  # some filings emit unprefixed


@dataclass(frozen=True)
class ParsedContext:
    context_id: str
    period_start: Optional[date]   # None if instant
    period_end: date               # equals the instant date for instant contexts
    dimensions: dict[str, str]     # {axis_qname: member_qname}, possibly empty

    @property
    def is_instant(self) -> bool:
        return self.period_start is None

    @property
    def dimensions_json(self) -> str:
        # Stable key order so identical dimensions produce identical strings,
        # which the PIT engine relies on for dedup.
        if not self.dimensions:
            return ""
        return json.dumps(self.dimensions, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class ParsedFact:
    concept: str            # "us-gaap:Revenues"
    taxonomy: str           # "us-gaap"
    value: float
    unit: str               # "USD"
    context: ParsedContext


def _parse_xbrl_date(s: str) -> date:
    # Sometimes "2020-12-31", sometimes "2020-12-31T00:00:00" or with offset.
    s = s.strip()
    try:
        return date.fromisoformat(s[:10])
    except ValueError:
        return datetime.fromisoformat(s).date()


def _iter_tag(root: etree._Element, tag_lower: str):
    """Yield descendants whose tag equals `tag_lower` (case-insensitive)."""
    for el in root.iter():
        if isinstance(el.tag, str) and el.tag.lower() == tag_lower:
            yield el


def _find_child(parent: etree._Element, tag_lower: str) -> Optional[etree._Element]:
    for child in parent:
        if isinstance(child.tag, str) and child.tag.lower() == tag_lower:
            return child
    return None


def _parse_contexts(root: etree._Element) -> dict[str, ParsedContext]:
    contexts: dict[str, ParsedContext] = {}
    for ctx in _iter_tag(root, TAG_XBRLI_CONTEXT):
        ctx_id = ctx.get("id")
        if ctx_id is None:
            continue

        period = _find_child(ctx, TAG_XBRLI_PERIOD)
        if period is None:
            continue
        instant = _find_child(period, TAG_XBRLI_INSTANT)
        if instant is not None:
            d = _parse_xbrl_date(instant.text or "")
            period_start, period_end = None, d
        else:
            sd = _find_child(period, TAG_XBRLI_START)
            ed = _find_child(period, TAG_XBRLI_END)
            if sd is None or ed is None:
                continue
            period_start = _parse_xbrl_date(sd.text or "")
            period_end = _parse_xbrl_date(ed.text or "")

        dims: dict[str, str] = {}
        for container_tag in (TAG_XBRLI_SEGMENT, TAG_XBRLI_SCENARIO):
            for parent in _iter_tag(ctx, container_tag):
                for em in _iter_tag(parent, TAG_XBRLDI_EXPLICIT):
                    axis = em.get("dimension")
                    member = (em.text or "").strip()
                    if axis and member:
                        dims[axis] = member

        contexts[ctx_id] = ParsedContext(
            context_id=ctx_id,
            period_start=period_start,
            period_end=period_end,
            dimensions=dims,
        )
    return contexts


_UNIT_MEASURE_RE = re.compile(r"^[^:]*:(?P<name>[A-Za-z0-9_]+)$")


def _parse_units(root: etree._Element) -> dict[str, str]:
    """Return {unit_id: simple_name}. Composite units (e.g. USD/share) become
    'USD_per_shares'. Best-effort: rare degenerate forms fall back to 'unknown'."""
    units: dict[str, str] = {}
    for u in _iter_tag(root, TAG_XBRLI_UNIT):
        uid = u.get("id")
        if uid is None:
            continue
        divide = _find_child(u, TAG_XBRLI_DIVIDE)
        if divide is not None:
            num = [m.text or "" for m in _iter_tag(divide, TAG_XBRLI_MEASURE)]
            # First half are numerator measures, second half denominator — but
            # we don't get that grouping cleanly without going through nested
            # unitNumerator/unitDenominator. For our purposes (USD-only mostly)
            # treat the first as num, last as den.
            if len(num) >= 2:
                units[uid] = f"{_simple_unit([num[0]])}_per_{_simple_unit([num[-1]])}"
            else:
                units[uid] = "unknown"
        else:
            measures = [m.text or "" for m in _iter_tag(u, TAG_XBRLI_MEASURE)]
            units[uid] = _simple_unit(measures)
    return units


def _simple_unit(measures: list[str]) -> str:
    if not measures:
        return "unknown"
    m = measures[0]
    match = _UNIT_MEASURE_RE.match(m)
    return match.group("name") if match else m


def _split_qname(qname: str) -> tuple[str, str]:
    """'us-gaap:Revenues' -> ('us-gaap', 'us-gaap:Revenues')."""
    if ":" not in qname:
        return ("", qname)
    return (qname.split(":", 1)[0], qname)


def parse_ixbrl(html_body: bytes) -> Iterator[ParsedFact]:
    """Yield all numeric XBRL facts from an iXBRL HTML primary document."""
    root = lxml_html.fromstring(html_body)
    contexts = _parse_contexts(root)
    units = _parse_units(root)
    if not contexts:
        return

    for el in root.iter():
        if not isinstance(el.tag, str):
            continue
        if el.tag.lower() not in TAG_IX_NONFRACTION:
            continue
        # xsi:nil="true" facts mean "no value reported" — skip.
        if (el.get(ATTR_NIL) or el.get(ATTR_NIL_LOWER)) == "true":
            continue
        # lxml HTML lowercases attributes.
        name = el.get("name")
        ctx_ref = el.get("contextref")
        unit_ref = el.get("unitref")
        if not (name and ctx_ref):
            continue
        ctx = contexts.get(ctx_ref)
        if ctx is None:
            continue

        raw = (el.text or "").replace(",", "").strip()
        if not raw:
            continue
        try:
            v = float(raw)
        except ValueError:
            continue

        # scale: power of 10 applied to the raw value
        scale = el.get("scale")
        if scale:
            try:
                v *= 10 ** int(scale)
            except ValueError:
                pass
        # sign: "-" inverts; XBRL stores absolute value in some filings
        if el.get("sign") == "-":
            v = -v
        # format hints (ixt:numdotdecimal etc.) — already handled by `float()`
        # because we stripped thousands separators above.

        unit = units.get(unit_ref, unit_ref or "unknown")
        taxonomy, concept = _split_qname(name)
        yield ParsedFact(
            concept=concept,
            taxonomy=taxonomy,
            value=v,
            unit=unit,
            context=ctx,
        )
