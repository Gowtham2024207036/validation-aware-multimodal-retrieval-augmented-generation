"""
arch5_query_expansion.py  —  Architecture 5: Query Expansion RAG
-----------------------------------------------------------------
Strategy: Rewrite the original query into N sub-queries, retrieve
candidates for each, then merge and re-rank using RRF.

Sub-query generation (rule-based, no LLM needed):
  1. Original query as-is
  2. Stripped to core financial metric + company + year
  3. Alternative phrasing using financial synonyms
  4. Entity-focused: just company + year + doc type
  5. Metric-focused: just the financial metric name

Why this helps:
  - A single query may miss relevant docs due to vocabulary mismatch
  - "Long-term Debt to Total Liabilities" might be stored as "leverage ratio"
    or "debt-to-assets" in different documents — expansion catches all variants
  - Multiple independent retrievals increase recall substantially

Strengths : Highest recall; handles ambiguous queries; vocabulary-robust
Weaknesses: Slower (N × retrieval calls); possible noise from irrelevant
            sub-queries; requires good synonym mapping for financial terms
"""

import re
from collections import defaultdict
from base_retriever import SharedModels, tokenize, deduplicate
import config

TOP_N      = 3
CANDIDATE  = 20
RRF_K      = 60

# Financial metric synonym map — expands the core metric term
METRIC_SYNONYMS = {
    "long-term debt":              ["long term debt", "LTD", "non-current debt", "long term borrowings"],
    "total liabilities":           ["total debt", "total obligations", "liabilities"],
    "gross profit margin":         ["gross margin", "gross profit %", "GPM", "gross profitability"],
    "revenue":                     ["sales", "net revenue", "top line", "turnover", "net sales"],
    "operating income":            ["EBIT", "operating profit", "operating earnings"],
    "net income":                  ["net profit", "bottom line", "earnings", "net earnings"],
    "earnings per share":          ["EPS", "diluted EPS", "basic EPS"],
    "return on equity":            ["ROE", "return on equity", "equity return"],
    "debt to equity":              ["D/E ratio", "leverage ratio", "gearing ratio"],
    "current ratio":               ["liquidity ratio", "current assets to current liabilities"],
    "capital expenditure":         ["capex", "capital spending", "PP&E additions"],
    "free cash flow":              ["FCF", "operating cash flow minus capex"],
    "total assets":                ["asset base", "total asset value"],
}

COMPANY_YEAR_RE = re.compile(r"\b(FY)?(\d{4})\b", re.IGNORECASE)
COMPANY_MAP     = {
    "costco": "COSTCO", "amazon": "AMAZON", "apple": "APPLE",
    "google": "GOOGL",  "alphabet": "GOOGL", "microsoft": "MSFT",
    "netflix": "NETFLIX", "tesla": "TSLA", "adobe": "ADOBE",
    "walmart": "WALMART", "nike": "NIKE",
}


def extract_core(query: str) -> tuple[str | None, str | None]:
    q_lower  = query.lower()
    company  = next((v for k, v in COMPANY_MAP.items() if k in q_lower), None)
    yr_match = COMPANY_YEAR_RE.search(query)
    year     = yr_match.group(2) if yr_match else None
    return company, year


def find_metric_synonyms(query: str) -> list[str]:
    """Find any metric in the query and return its synonym list."""
    q_lower = query.lower()
    for metric, synonyms in METRIC_SYNONYMS.items():
        if metric in q_lower:
            return synonyms
    return []


def generate_sub_queries(query: str) -> list[str]:
    """
    Generate up to 5 diverse sub-queries from the original.
    Each targets a different aspect of the information need.
    """
    company, year = extract_core(query)
    synonyms      = find_metric_synonyms(query)

    sub_queries = [query]  # always include original

    # Sub-query 2: entity-focused (company + year + annual report)
    if company and year:
        sub_queries.append(f"{company} {year} annual report 10K financial statements")

    # Sub-query 3: metric synonym variants
    for syn in synonyms[:2]:  # max 2 synonym variants
        sub_q = re.sub(
            '|'.join(re.escape(k) for k in METRIC_SYNONYMS),
            syn, query, flags=re.IGNORECASE, count=1
        )
        if sub_q != query:
            sub_queries.append(sub_q)
        else:
            # Couldn't substitute — just append synonym as extra context
            if company and year:
                sub_queries.append(f"{company} {year} {syn}")

    # Sub-query 4: metric-only (strips company/year to find generic definitions)
    if synonyms:
        sub_queries.append(f"{synonyms[0]} financial ratio definition calculation")

    # Sub-query 5: balance sheet / income statement context
    if company and year:
        sub_queries.append(f"{company} {year} balance sheet income statement")

    # Deduplicate while preserving order
    seen, unique = set(), []
    for q in sub_queries:
        if q not in seen:
            seen.add(q)
            unique.append(q)

    return unique[:5]  # cap at 5


def rrf_fuse(ranked_lists: list[list[dict]], k: int = RRF_K) -> list[dict]:
    scores    = defaultdict(float)
    id_to_doc = {}
    for lst in ranked_lists:
        for rank, doc in enumerate(lst, start=1):
            scores[doc["id"]] += 1.0 / (k + rank)
            if doc["id"] not in id_to_doc:
                id_to_doc[doc["id"]] = doc
    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    out = []
    for did in sorted_ids:
        item = id_to_doc[did].copy()
        item["rrf_score"] = scores[did]
        out.append(item)
    return out


def retrieve(query: str, models: SharedModels) -> dict:
    sub_queries = generate_sub_queries(query)
    print(f"  [Arch5] {len(sub_queries)} sub-queries generated:")
    for i, sq in enumerate(sub_queries, 1):
        print(f"    {i}. {sq}")

    all_text_lists  = []
    all_image_lists = []

    for sq in sub_queries:
        text_vec = models.encode_text(sq)
        clip_vec = models.encode_clip(sq)

        try:
            tr = models.client.query_points(
                collection_name=config.TEXT_COLLECTION,
                query=text_vec, limit=CANDIDATE,
            )
            all_text_lists.append(
                [{"id": p.id, "score": p.score, "sub_query": sq, **p.payload}
                 for p in tr.points]
            )
        except Exception:
            pass

        try:
            ir = models.client.query_points(
                collection_name=config.IMAGE_COLLECTION,
                query=clip_vec, limit=CANDIDATE,
            )
            all_image_lists.append(
                [{"id": p.id, "score": p.score, "sub_query": sq, **p.payload}
                 for p in ir.points]
            )
        except Exception:
            pass

    fused_text  = rrf_fuse(all_text_lists)  if all_text_lists  else []
    fused_image = rrf_fuse(all_image_lists) if all_image_lists else []

    return {
        "text":        deduplicate(fused_text,  TOP_N),
        "image":       deduplicate(fused_image, TOP_N),
        "sub_queries": sub_queries,
    }
