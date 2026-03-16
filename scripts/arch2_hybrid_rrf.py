"""
arch2_hybrid_rrf.py  —  Architecture 2: Hybrid RAG with RRF
------------------------------------------------------------
Strategy: BM25 + Dense vector search fused via Reciprocal Rank Fusion.
  - BM25 uses expanded query (ticker aliases appended)
  - Dense vector uses original query
  - RRF merges: score = 1 / (RRF_K + rank), summed across both lists
  - Documents appearing in BOTH lists get double-boosted

Strengths : Entity-aware (BM25) + semantic (vector) + parameter-free fusion
Weaknesses: BM25 can still be dominated by generic financial terms;
            wrong-company results persist when semantic signal is very strong
"""

from collections import defaultdict
from base_retriever import SharedModels, tokenize, deduplicate
import config

TOP_N      = 3
CANDIDATE  = 50
RRF_K      = 60

# Company → ticker expansion map
COMPANY_ALIASES = {
    "costco":    "COST Costco Wholesale annual report 10K",
    "amazon":    "AMZN Amazon annual report 10K",
    "apple":     "AAPL Apple Inc annual report 10K",
    "google":    "GOOGL Alphabet annual report 10K",
    "microsoft": "MSFT Microsoft annual report 10K",
    "netflix":   "NFLX Netflix annual report 10K",
    "tesla":     "TSLA Tesla annual report 10K",
    "adobe":     "ADBE Adobe annual report 10K",
    "walmart":   "WMT Walmart annual report 10K",
    "nike":      "NKE Nike annual report 10K",
}


def expand_query(query: str) -> str:
    q_lower    = query.lower()
    expansions = [exp for kw, exp in COMPANY_ALIASES.items() if kw in q_lower]
    return (query + " " + " ".join(expansions)).strip() if expansions else query


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


def _bm25_candidates(bm25, records, query_tokens, n):
    import numpy as np
    scores   = bm25.get_scores(query_tokens)
    top_idx  = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
    return [{"id": records[i]["id"], "bm25_score": float(scores[i]),
             **records[i]["payload"]} for i in top_idx]


def retrieve(query: str, models: SharedModels) -> dict:
    expanded = expand_query(query)

    # ---------- TEXT ----------
    bm25_text = _bm25_candidates(
        models.text_bm25, models.text_records, tokenize(expanded), CANDIDATE
    )
    try:
        tr = models.client.query_points(
            collection_name=config.TEXT_COLLECTION,
            query=models.encode_text(query), limit=CANDIDATE,
        )
        vec_text = [{"id": p.id, "vector_score": p.score, **p.payload} for p in tr.points]
    except Exception:
        vec_text = []

    fused_text = rrf_fuse([bm25_text, vec_text])

    # ---------- IMAGE ----------
    bm25_img = _bm25_candidates(
        models.image_bm25, models.image_records, tokenize(expanded), CANDIDATE
    )
    try:
        ir = models.client.query_points(
            collection_name=config.IMAGE_COLLECTION,
            query=models.encode_clip(query), limit=CANDIDATE,
        )
        vec_img = [{"id": p.id, "vector_score": p.score, **p.payload} for p in ir.points]
    except Exception:
        vec_img = []

    fused_img = rrf_fuse([bm25_img, vec_img])

    return {
        "text":  deduplicate(fused_text, TOP_N),
        "image": deduplicate(fused_img,  TOP_N),
    }
