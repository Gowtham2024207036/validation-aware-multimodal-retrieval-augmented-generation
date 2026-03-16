"""
arch3_metadata_filter.py  —  Architecture 3: Metadata-Filtered RAG
--------------------------------------------------------------------
Strategy: Extract structured entities from the query, apply as a hard
Qdrant metadata pre-filter BEFORE vector search. Only documents matching
the filter are searched — dramatically narrows the candidate space.

Filter logic:
  1. Extract company name → map to doc_name prefix (e.g. "COSTCO" → "COSTCO")
  2. Extract fiscal year  → map to year token  (e.g. "FY2021" → "2021")
  3. Build Qdrant Filter: doc_name must contain company AND year substring
  4. Fall back to unfiltered search if no entities found

Strengths : Highest precision for entity-specific queries; no wrong-company results
Weaknesses: Requires entity recognition; fails on vague/generic queries;
            doc_name must contain company/year in a predictable format
"""

import re
from base_retriever import SharedModels, deduplicate
from qdrant_client.http import models as qmodels
import config

TOP_N     = 3
CANDIDATE = 20

# Company name → doc_name keyword mapping
COMPANY_MAP = {
    "costco":    "COSTCO",
    "amazon":    "AMAZON",
    "apple":     "APPLE",
    "alphabet":  "GOOGL",
    "google":    "GOOGL",
    "microsoft": "MSFT",
    "netflix":   "NETFLIX",
    "tesla":     "TSLA",
    "adobe":     "ADOBE",
    "walmart":   "WALMART",
    "nike":      "NIKE",
    "bestbuy":   "BESTBUY",
    "best buy":  "BESTBUY",
}

YEAR_RE = re.compile(r"\b(FY)?(\d{4})\b", re.IGNORECASE)


def extract_entities(query: str) -> tuple[str | None, str | None]:
    """Returns (company_keyword, year_string) or (None, None)."""
    q_lower  = query.lower()
    company  = next((v for k, v in COMPANY_MAP.items() if k in q_lower), None)

    year_match = YEAR_RE.search(query)
    year       = year_match.group(2) if year_match else None

    return company, year


def build_filter(company: str | None, year: str | None):
    """Build a Qdrant must-match filter on the doc_name payload field."""
    conditions = []
    if company:
        conditions.append(
            qmodels.FieldCondition(
                key="doc_name",
                match=qmodels.MatchText(text=company),  # substring match
            )
        )
    if year:
        conditions.append(
            qmodels.FieldCondition(
                key="doc_name",
                match=qmodels.MatchText(text=year),
            )
        )
    if not conditions:
        return None
    return qmodels.Filter(must=conditions)


def retrieve(query: str, models: SharedModels) -> dict:
    company, year = extract_entities(query)
    q_filter      = build_filter(company, year)

    if q_filter:
        print(f"  [Arch3] Filter: company={company}, year={year}")
    else:
        print(f"  [Arch3] No entities found — unfiltered search")

    text_vec = models.encode_text(query)
    clip_vec = models.encode_clip(query)

    # --- Text collection with filter ---
    try:
        tr = models.client.query_points(
            collection_name=config.TEXT_COLLECTION,
            query=text_vec,
            query_filter=q_filter,   # None = no filter
            limit=CANDIDATE,
        )
        text_hits = [{"id": p.id, "score": p.score, **p.payload} for p in tr.points]
    except Exception as e:
        print(f"  [Arch3] Text search error: {e}")
        text_hits = []

    # --- Image collection with filter ---
    try:
        ir = models.client.query_points(
            collection_name=config.IMAGE_COLLECTION,
            query=clip_vec,
            query_filter=q_filter,
            limit=CANDIDATE,
        )
        image_hits = [{"id": p.id, "score": p.score, **p.payload} for p in ir.points]
    except Exception as e:
        print(f"  [Arch3] Image search error: {e}")
        image_hits = []

    # If filter produced zero results, fall back to unfiltered
    if not text_hits and q_filter:
        print(f"  [Arch3] Filter returned 0 text results — falling back to unfiltered")
        try:
            tr = models.client.query_points(
                collection_name=config.TEXT_COLLECTION,
                query=text_vec, limit=CANDIDATE,
            )
            text_hits = [{"id": p.id, "score": p.score, **p.payload} for p in tr.points]
        except Exception:
            pass

    if not image_hits and q_filter:
        print(f"  [Arch3] Filter returned 0 image results — falling back to unfiltered")
        try:
            ir = models.client.query_points(
                collection_name=config.IMAGE_COLLECTION,
                query=clip_vec, limit=CANDIDATE,
            )
            image_hits = [{"id": p.id, "score": p.score, **p.payload} for p in ir.points]
        except Exception:
            pass

    return {
        "text":  deduplicate(text_hits,  TOP_N),
        "image": deduplicate(image_hits, TOP_N),
    }
