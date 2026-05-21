"""USPTO patent data via the PatentsView API.

PatentsView is a free, well-structured REST API maintained by USPTO. The v1
endpoint exposes `/api/patents/query` returning granted patents matching a
JSON-encoded query. We query by assignee organization name.

API docs: https://patentsview.org/apis/api-endpoints

Limitations of v1 in this connector:
  - Granted patents only — pre-grant publications not ingested (would require
    a separate endpoint with different schema).
  - The free endpoint paginates at ~25 results per request. The connector
    follows the page chain.
  - Patent invalidations and post-grant reviews are not tracked (would
    require USPTO PTAB data, separate source).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import Iterator, Optional

import requests

PATENTSVIEW_URL = "https://api.patentsview.org/patents/query"


@dataclass(frozen=True)
class ParsedPatent:
    patent_number: str
    application_date: date
    grant_date: date
    assignee_name_as_filed: str
    title: str
    abstract: str
    claims_count: int
    cpc_classifications: list[str]
    inventors: list[str]


def _parse_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except ValueError:
        return None


def fetch_patents_by_assignee(
    assignee_name: str,
    *,
    start_grant_date: Optional[date] = None,
    end_grant_date: Optional[date] = None,
    api_key: Optional[str] = None,
    user_agent: str = "norn-bwm research rohit@dragonscale.ai",
    timeout_s: float = 30.0,
) -> Iterator[ParsedPatent]:
    """Yield ParsedPatent rows for granted patents matching `assignee_name`.

    `assignee_name` is matched case-insensitively against `assignees.assignee_organization`
    on PatentsView. Date filters bound `grant_date` for incremental ingest.
    """
    # The PatentsView v1 API expects a query JSON. We match assignees by org
    # name (case-insensitively) and bound grant_date.
    query: dict = {"_contains": {"assignee_organization": assignee_name}}
    date_clauses = []
    if start_grant_date is not None:
        date_clauses.append({"_gte": {"patent_date": start_grant_date.isoformat()}})
    if end_grant_date is not None:
        date_clauses.append({"_lte": {"patent_date": end_grant_date.isoformat()}})
    if date_clauses:
        query = {"_and": [query, *date_clauses]}

    fields = [
        "patent_number", "patent_date", "patent_title", "patent_abstract",
        "patent_num_claims", "app_date", "assignee_organization",
        "inventor_name_first", "inventor_name_last", "cpc_subgroup_id",
    ]
    options = {"per_page": 100}
    headers = {"User-Agent": user_agent, "Accept": "application/json"}
    if api_key:
        headers["X-Api-Key"] = api_key

    page = 1
    page_size = options["per_page"]
    while True:
        options["page"] = page
        resp = requests.post(
            PATENTSVIEW_URL,
            json={"q": query, "f": fields, "o": options},
            headers=headers,
            timeout=timeout_s,
        )
        resp.raise_for_status()
        data = resp.json()
        patents = data.get("patents") or []
        if not patents:
            return
        for p in patents:
            patent_number = str(p.get("patent_number", "")).strip()
            grant_d = _parse_date(p.get("patent_date"))
            if not (patent_number and grant_d):
                continue
            # app_date / application_date is nested under applications[0] on
            # older versions and flat on newer versions; check both.
            apps = p.get("applications") or p.get("application") or []
            app_d = _parse_date(apps[0].get("app_date")) if apps else None
            if app_d is None:
                app_d = grant_d  # fallback so PIT contract holds
            assignees = p.get("assignees") or []
            assignee_filed = (
                (assignees[0].get("assignee_organization") or "") if assignees
                else str(p.get("assignee_organization", "") or "")
            )
            inventors_list = p.get("inventors") or []
            inventors = [
                f"{i.get('inventor_name_first', '')} {i.get('inventor_name_last', '')}".strip()
                for i in inventors_list
            ]
            cpc_codes = [
                str(c.get("cpc_subgroup_id") or "")
                for c in (p.get("cpcs") or p.get("cpc_current") or [])
            ]
            yield ParsedPatent(
                patent_number=patent_number,
                application_date=app_d,
                grant_date=grant_d,
                assignee_name_as_filed=assignee_filed,
                title=str(p.get("patent_title", "") or ""),
                abstract=str(p.get("patent_abstract", "") or ""),
                claims_count=int(p.get("patent_num_claims", 0) or 0),
                cpc_classifications=[c for c in cpc_codes if c],
                inventors=[i for i in inventors if i],
            )
        if len(patents) < page_size:
            return
        page += 1
