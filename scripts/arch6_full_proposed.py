"""
arch6_full_proposed.py  —  Architecture 6: Full Proposed Architecture
-----------------------------------------------------------------------
This is YOUR proposed architecture from the diagram — the full system
combining all components:

  1. Query Processing (type detection + contextual enhancement)
  2. Hybrid Retrieval (FAISS-equivalent via Qdrant + BM25 + Score Fusion)
  3. Context Quality Analyzer (cosine similarity + noise estimation)
  4. Context Validation Model (DistilBERT classifier → label 0/1/2)
  5. Context Decision Engine (4-stage validation pipeline)
  6. Validation-Aware Context Re-Ranking (LLM-based confidence scoring)
  7. Question Answering Module (Qwen2.5-VL via LM Studio)
  8. Answer Post-Processing (citation parsing + confidence score)

This architecture is compared against Architectures 1-5 in your evaluation.
The key novelty: the CDE validation layer between retrieval and generation.

Strengths:  Highest precision — removes irrelevant/redundant chunks before LLM
            Modality-aware — routes numeric queries toward table/image chunks
            Validation-scored — each chunk has an interpretable confidence score
Weaknesses: Slowest — requires N forward passes through DistilBERT per query
            Depends on trained CDE model quality
"""

from collections import defaultdict
from base_retriever import SharedModels, tokenize, deduplicate
from context_decision_engine import ContextDecisionEngine
import config

TOP_N     = 3
CANDIDATE = 50
RRF_K     = 60

# Lazy-load CDE (shared across calls)
_cde_instance = None

def get_cde() -> ContextDecisionEngine:
    global _cde_instance
    if _cde_instance is None:
        _cde_instance = ContextDecisionEngine()
    return _cde_instance


# Company alias map (same as arch2)
COMPANY_ALIASES = {
    "costco":    "COST Costco Wholesale annual report 10K",
    "amazon":    "AMZN Amazon annual report 10K",
    "apple":     "AAPL Apple Inc annual report 10K",
    "google":    "GOOGL Alphabet annual report 10K",
    "microsoft": "MSFT Microsoft annual report 10K",
    "netflix":   "NFLX Netflix annual report 10K",
    "tesla":     "TSLA Tesla annual report 10K",
    "adobe":     "ADBE Adobe annual report 10K",
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
    scores  = bm25.get_scores(query_tokens)
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
    return [{"id": records[i]["id"], "bm25_score": float(scores[i]),
             **records[i]["payload"]} for i in top_idx]


def retrieve(query: str, models: SharedModels) -> dict:
    expanded = expand_query(query)

    # ── Stage 1: Hybrid Retrieval (same as arch2) ──────────────────
    bm25_text = _bm25_candidates(
        models.text_bm25, models.text_records, tokenize(expanded), CANDIDATE
    )
    try:
        tr = models.client.query_points(
            collection_name=config.TEXT_COLLECTION,
            query=models.encode_text(query), limit=CANDIDATE,
        )
        # Deduplicate vector results
        seen, vec_text = set(), []
        for p in tr.points:
            if p.id not in seen:
                seen.add(p.id)
                vec_text.append({"id": p.id, "vector_score": p.score, **p.payload})
    except Exception:
        vec_text = []

    bm25_img = _bm25_candidates(
        models.image_bm25, models.image_records, tokenize(expanded), CANDIDATE
    )
    try:
        ir = models.client.query_points(
            collection_name=config.IMAGE_COLLECTION,
            query=models.encode_clip(query), limit=CANDIDATE,
        )
        seen, vec_img = set(), []
        for p in ir.points:
            if p.id not in seen:
                seen.add(p.id)
                vec_img.append({"id": p.id, "vector_score": p.score, **p.payload})
    except Exception:
        vec_img = []

    # RRF fusion — get top candidates before CDE validation
    fused_text  = rrf_fuse([bm25_text, vec_text])[:CANDIDATE]
    fused_image = rrf_fuse([bm25_img,  vec_img ])[:CANDIDATE]

    # ── Stage 2: Context Decision Engine (CDE) ─────────────────────
    cde = get_cde()
    validated = cde.validate(
        query      = query,
        text_hits  = fused_text,
        image_hits = fused_image,
        top_k      = TOP_N,
    )

    # Final deduplication by content (same as other architectures)
    return {
        "text":  deduplicate(validated["text"],  TOP_N),
        "image": deduplicate(validated["image"], TOP_N),
    }