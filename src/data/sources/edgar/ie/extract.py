"""LLM-based extraction of supply-chain / business-relationship edges from
10-K narrative text, via Azure OpenAI.

Why an LLM and not a fine-tuned NER model? The relations of interest
(supplier, customer, competitor, partner, subsidiary) are conveyed in
running prose with substantial linguistic variation — "our largest customer
accounts for…", "we rely on a single supplier of…", "we compete primarily
with…", "wholly-owned subsidiary in…". Per-relation classifiers exist but
require labeled corpora we don't have. A capable LLM with a structured-
output prompt gets us to a working v0 quickly; we can later distill into
a smaller model when we have labels from production usage.

### Backend: Azure OpenAI

The `AzureOpenAI` client from the `openai` package targets an Azure-hosted
deployment. Configuration via env vars:

  AZURE_OPENAI_ENDPOINT     https://{resource}.openai.azure.com/
  AZURE_OPENAI_API_KEY      api key from the Azure portal
  AZURE_OPENAI_API_VERSION  defaults to "2024-10-21" (Azure-stable)
  BWM_IE_DEPLOYMENT         deployment name (defaults to "gpt-5.5")

GPT-5-family models on Azure use `max_completion_tokens` (not `max_tokens`)
and may restrict `temperature` to the default. We pass `temperature=0` for
deterministic extraction; if the deployment rejects it, the wrapper falls
back to omitting the param.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Optional

# Imported lazily inside _client_lazy so tests that don't need a live LLM
# don't pay the import cost or require env vars to be set.

DEFAULT_DEPLOYMENT = os.environ.get("BWM_IE_DEPLOYMENT", "gpt-5.5")
DEFAULT_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21")
DEFAULT_MAX_COMPLETION_TOKENS = 1024

EXTRACT_SYSTEM = """You extract structured business-relationship edges from \
public-company SEC filing text. Output ONLY a JSON array — no prose, no \
markdown fencing. Each element has the shape:

  {
    "target_name": "string — counterparty name as filed",
    "relationship_type": "one of: customer, supplier, competitor, partner, subsidiary, lender, regulator, other",
    "weight_qualifier": "string — optional qualifier like '10% customer', 'sole supplier'",
    "evidence_span": "string — verbatim quote of ~200 chars supporting this edge",
    "confidence": 0.0-1.0
  }

Rules:
  - One edge per distinct (counterparty, relationship_type) pair.
  - Skip generic mentions ("our suppliers"). Only emit named counterparties.
  - The viewer of this text IS the issuer (the company filing); edges are FROM issuer TO target.
  - If text mentions concentration ("one customer represents 10% of revenue") without naming the customer, do NOT emit an edge.
  - If the input text contains no extractable edges, return [].
"""

EXTRACT_USER_TEMPLATE = """Issuer (the company filing this 10-K): {issuer_name}

Item {item_id} excerpt:
\"\"\"
{text}
\"\"\"

Return the JSON array now."""


@dataclass(frozen=True)
class RawEdge:
    target_name: str
    relationship_type: str
    weight_qualifier: str
    evidence_span: str
    confidence: float


def _evidence_hash(span: str) -> str:
    return hashlib.sha1(span.encode("utf-8", errors="replace")).hexdigest()[:12]


def parse_response(content: str) -> list[RawEdge]:
    """Extract a JSON array from the model response, tolerating leading/trailing slop.

    Handles common deviations from the prompt:
      - leading/trailing prose ("Here are the edges: [...]")
      - markdown code fences (```json ... ```)
      - empty arrays, malformed elements, wrong outer type
    """
    if not content:
        return []
    # Strip markdown fences if present
    cleaned = content.strip()
    if cleaned.startswith("```"):
        # Drop everything up to the first newline of the fence and the trailing fence
        lines = cleaned.split("\n")
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines)
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start < 0 or end < 0 or end <= start:
        return []
    blob = cleaned[start : end + 1]
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    edges: list[RawEdge] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        target = str(item.get("target_name", "")).strip()
        rel = str(item.get("relationship_type", "")).strip().lower()
        evidence = str(item.get("evidence_span", "")).strip()
        if not (target and rel and evidence):
            continue
        try:
            conf = float(item.get("confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5
        edges.append(
            RawEdge(
                target_name=target,
                relationship_type=rel,
                weight_qualifier=str(item.get("weight_qualifier", "")).strip(),
                evidence_span=evidence,
                confidence=conf,
            )
        )
    return edges


class EdgeExtractor:
    """Azure-OpenAI-backed extractor for supply-chain edges.

    The client is constructed lazily on first call so tests that exercise
    `parse_response` standalone don't need Azure credentials. Each call to
    `extract()` is one Chat Completions request.
    """

    def __init__(self, deployment: Optional[str] = None) -> None:
        self.deployment = deployment or DEFAULT_DEPLOYMENT
        self._client = None  # type: ignore[assignment]
        # Use the deployment name as the "model" string for cost-tracking
        # purposes; the actual model identity is whatever Azure has deployed
        # under this name (likely gpt-5.5 family per user spec).
        self.model = self.deployment

    def _client_lazy(self):
        if self._client is None:
            from openai import AzureOpenAI

            endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
            api_key = os.environ.get("AZURE_OPENAI_API_KEY")
            if not endpoint or not api_key:
                raise RuntimeError(
                    "Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY env vars "
                    "(and AZURE_OPENAI_API_VERSION; defaults to "
                    f"{DEFAULT_API_VERSION!r})."
                )
            self._client = AzureOpenAI(
                api_key=api_key,
                api_version=os.environ.get("AZURE_OPENAI_API_VERSION", DEFAULT_API_VERSION),
                azure_endpoint=endpoint,
            )
        return self._client

    def extract(
        self,
        issuer_name: str,
        item_id: str,
        text: str,
    ) -> list[RawEdge]:
        client = self._client_lazy()
        messages = [
            {"role": "system", "content": EXTRACT_SYSTEM},
            {
                "role": "user",
                "content": EXTRACT_USER_TEMPLATE.format(
                    issuer_name=issuer_name,
                    item_id=item_id,
                    text=text[:8000],  # cap to keep cost predictable
                ),
            },
        ]
        # GPT-5 family on Azure uses max_completion_tokens; older models took
        # max_tokens. Try the new name first, fall back if rejected.
        try:
            resp = client.chat.completions.create(
                model=self.deployment,
                messages=messages,
                max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
                temperature=0,
            )
        except TypeError:
            # SDK doesn't know max_completion_tokens (older SDK) — use max_tokens
            resp = client.chat.completions.create(
                model=self.deployment,
                messages=messages,
                max_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
                temperature=0,
            )
        except Exception as e:  # noqa: BLE001
            # Some GPT-5 deployments reject `temperature` other than default.
            # Retry without it.
            if "temperature" in str(e).lower():
                resp = client.chat.completions.create(
                    model=self.deployment,
                    messages=messages,
                    max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
                )
            else:
                raise
        content = resp.choices[0].message.content or ""
        return parse_response(content)
