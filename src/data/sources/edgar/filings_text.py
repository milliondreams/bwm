"""Filing-text extraction: HTML primary doc → per-Item clean chunks.

10-K / 10-Q / 8-K filings are HTML documents (modern filings are iXBRL but the
narrative content is regular text). We:

  1. Strip iXBRL inline markup and HTML structure, preserving block-level
     boundaries so paragraph structure survives.
  2. Identify Item boundaries via regex on "Item N." patterns. For each form
     class, we know which Items are expected (10-K has 1, 1A, 1B, 2, 3, 4,
     5, 6, 7, 7A, 8, 9, 9A, 9B, 10, 11, 12, 13, 14, 15, 16). 8-K Items use a
     decimal form (1.01, 2.02, etc.).
  3. For each Item span, chunk the text into ~1000-token windows with 200-
     token overlap. Tokens are approximated as ~4 chars/token which is
     within 15% for English; precise tokenizer happens later when we embed.

The output is deterministic across runs for the same input bytes so we can
re-extract without producing diff churn — important for Stage 5's resumable
backfill.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterator

from lxml import html as lxml_html


# Approximate token count: 4 chars/token in English filings, empirically.
CHARS_PER_TOKEN = 4
CHUNK_TOKENS = 1000
CHUNK_OVERLAP_TOKENS = 200
CHUNK_CHARS = CHUNK_TOKENS * CHARS_PER_TOKEN
OVERLAP_CHARS = CHUNK_OVERLAP_TOKENS * CHARS_PER_TOKEN

# Item heading patterns. Both forms appear in the wild:
#   "Item 1. Business"            (10-K/Q standard)
#   "ITEM 1A. RISK FACTORS"        (caps variant)
#   "Item 1.01"                    (8-K decimal form)
# Greedy `.+?` consumes the title up to the next double-newline.
_ITEM_10KQ_RE = re.compile(
    r"(?im)^\s*item\s+(?P<id>\d{1,2}[A-Z]?)\.?\s+(?P<title>[^\n]{0,160})",
)
_ITEM_8K_RE = re.compile(
    r"(?im)^\s*item\s+(?P<id>\d+\.\d{2})\.?\s+(?P<title>[^\n]{0,160})",
)


@dataclass(frozen=True)
class ExtractedItem:
    item_id: str
    item_title: str
    text: str


@dataclass(frozen=True)
class TextChunk:
    item_id: str
    item_title: str
    chunk_idx: int
    n_chunks_in_item: int
    text: str


# ---------- HTML → plain text ------------------------------------------------


_BLOCK_TAGS = frozenset({
    "p", "div", "br", "li", "tr", "td", "th", "h1", "h2", "h3", "h4", "h5", "h6",
    "section", "article", "ul", "ol", "table", "header", "footer", "nav",
    "title", "blockquote",
})

# iXBRL inline tags (these wrap facts but we want to strip them, keeping inner text).
_IXBRL_TAGS = frozenset({
    "ix:header", "ix:hidden", "ix:resources", "ix:references", "ix:relationship",
    # We keep the *text content* of ix:nonFraction / ix:nonNumeric because those
    # carry visible numeric/textual content the reader sees. Only the structural
    # iXBRL containers above are pruned.
})


def html_to_text(html_body: bytes) -> str:
    """Convert iXBRL/HTML to clean plain text, preserving paragraph breaks.

    Approach:
      - Parse with lxml HTML parser.
      - Drop entire subtrees for elements that are not visible content
        (script, style, iXBRL header/hidden/resources blocks).
      - Walk the tree producing text with newlines inserted at block-element
        boundaries.
    """
    root = lxml_html.fromstring(html_body)

    # Drop non-visible subtrees.
    for tag in ("script", "style", "noscript"):
        for el in root.iter(tag):
            el.getparent().remove(el) if el.getparent() is not None else None
    # iXBRL containers: tag names are lowercased by the HTML parser.
    for el in list(root.iter()):
        if not isinstance(el.tag, str):
            continue
        if el.tag.lower() in _IXBRL_TAGS:
            parent = el.getparent()
            if parent is not None:
                parent.remove(el)

    out: list[str] = []
    _walk(root, out)
    text = "".join(out)

    # Normalize Unicode whitespace (NBSP etc.) and collapse runs.
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _walk(el, out: list[str]) -> None:
    tag = el.tag.lower() if isinstance(el.tag, str) else ""
    is_block = tag in _BLOCK_TAGS
    if is_block and out and not out[-1].endswith("\n"):
        out.append("\n")
    if el.text:
        out.append(el.text)
    for child in el:
        _walk(child, out)
        if child.tail:
            out.append(child.tail)
    if is_block:
        out.append("\n")


# ---------- Item segmentation ------------------------------------------------


def extract_items(text: str, form: str) -> list[ExtractedItem]:
    """Split a plain-text filing into ExtractedItem records.

    For 10-K/10-Q we use the integer+letter pattern (1, 1A, 7A...). For 8-K
    we use the decimal form (1.01, 2.02, ...). For other forms we fall back
    to the 10-K pattern, which catches most cases.
    """
    if form.startswith("8-K"):
        pattern = _ITEM_8K_RE
    else:
        pattern = _ITEM_10KQ_RE

    matches = list(pattern.finditer(text))
    if not matches:
        return []

    items: list[ExtractedItem] = []
    for i, m in enumerate(matches):
        item_id = m.group("id").upper()
        item_title = m.group("title").strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if not body:
            continue
        items.append(
            ExtractedItem(item_id=item_id, item_title=item_title, text=body)
        )

    # 10-K table-of-contents anti-pattern: many filings have a TOC at the top
    # which makes every Item match twice — once as a TOC entry (short body)
    # and once as the real section. Collapse by keeping the *longest* body
    # per item_id, which is the actual section.
    by_id: dict[str, ExtractedItem] = {}
    for item in items:
        existing = by_id.get(item.item_id)
        if existing is None or len(item.text) > len(existing.text):
            by_id[item.item_id] = item
    return [by_id[k] for k in sorted(by_id.keys())]


# ---------- Chunking ---------------------------------------------------------


def chunk_text(text: str) -> list[str]:
    """Sliding window over `text` producing fixed-size chunks with overlap.

    Boundaries align to whitespace where possible so we don't bisect a word.
    Deterministic: same input bytes always produce identical chunks.
    """
    if not text:
        return []
    if len(text) <= CHUNK_CHARS:
        return [text]

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + CHUNK_CHARS, n)
        # If we're not at the end, prefer a whitespace boundary nearby.
        if end < n:
            ws = text.rfind(" ", start + (CHUNK_CHARS - 200), end)
            if ws > start:
                end = ws
        chunks.append(text[start:end].strip())
        if end >= n:
            break
        start = max(end - OVERLAP_CHARS, start + 1)
    return [c for c in chunks if c]


def extract_chunks(html_body: bytes, form: str) -> Iterator[TextChunk]:
    """End-to-end: HTML → text → items → chunks."""
    text = html_to_text(html_body)
    items = extract_items(text, form)
    for item in items:
        chunks = chunk_text(item.text)
        total = len(chunks)
        for idx, chunk in enumerate(chunks):
            yield TextChunk(
                item_id=item.item_id,
                item_title=item.item_title,
                chunk_idx=idx,
                n_chunks_in_item=total,
                text=chunk,
            )
