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

TOP_N     = 5
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



KNOWN_COMPANIES = {
    "costco": "COSTCO", "amazon": "AMAZON", "apple": "APPLE",
    "netflix": "NETFLIX", "tesla": "TESLA", "microsoft": "MICROSOFT",
    "google": "GOOGLE", "alphabet": "GOOGLE", "walmart": "WALMART",
    "bestbuy": "BESTBUY", "best buy": "BESTBUY", "target": "TARGET",
    "nike": "NIKE", "adobe": "ADOBE", "salesforce": "SALESFORCE",
    "inditex": "INDITEX", "nestle": "NSRGY",
}

# Education / government document keyword expansion
# When query contains these terms, expand to help BM25 find the right doc
EDUCATION_EXPANSIONS = {
    "scholarship": "scholarship fee waiver tuition concession financial aid",
    "tnea": "TNEA tamil nadu engineering admissions counseling",
    "sc st": "scheduled caste scheduled tribe SC SCA ST reservation",
    "sc/st": "scheduled caste scheduled tribe SC SCA ST reservation",
    "sc ": "scheduled caste SC reservation community",
    "obc": "other backward class OBC BC MBC reservation",
    "admission": "admission eligibility counseling rank merit list",
    "cutoff": "cutoff rank merit list score opening closing",
    "eligibility": "eligibility qualification marks percentage criteria",
    "reservation": "reservation quota community category seats",
    "anna university": "Anna University CEG MIT Guindy engineering",
}

def _expand_education_query(query: str) -> str:
    """Expand education/government queries with domain-specific terms."""
    q_lower = query.lower()
    expansions = []
    for keyword, expansion in EDUCATION_EXPANSIONS.items():
        if keyword in q_lower:
            expansions.append(expansion)
    if expansions:
        return query + " " + " ".join(expansions)
    return query

def _extract_company(query: str) -> str:
    """Return uppercase company key if found in query, else empty string."""
    q = query.lower()
    for kw, code in KNOWN_COMPANIES.items():
        if kw in q:
            return code
    return ""


def _direct_lookup_by_docname(query: str, client, top_n: int = TOP_N) -> tuple[list, list]:
    """
    If query mentions a specific doc_name stored in Qdrant,
    fetch those chunks directly using a scroll filter — bypasses CLIP/BM25.
    Returns (text_hits, image_hits) or ([], []) if no doc_name match found.
    """
    from qdrant_client.http import models as qmodels
    import re

    # Extract candidate doc_name tokens from query (uppercase identifiers)
    candidates = re.findall(r'[A-Z][A-Z0-9_\-]{4,}', query.upper())

    # Also check for common document name patterns with numbers/underscores
    # e.g. "2_INFORMATION_BROCHURE" starts with digit
    digit_candidates = re.findall(r'\d+_[A-Z][A-Z0-9_]{3,}', query.upper())
    candidates = list(set(candidates + digit_candidates))

    if not candidates:
        return [], []

    text_hits  = []
    image_hits = []

    for candidate in candidates:
        # Try text collection
        try:
            results, _ = client.scroll(
                collection_name=config.TEXT_COLLECTION,
                scroll_filter=qmodels.Filter(
                    must=[qmodels.FieldCondition(
                        key="doc_name",
                        match=qmodels.MatchValue(value=candidate)
                    )]
                ),
                limit=top_n,
                with_payload=True,
                with_vectors=False,
            )
            for p in results:
                hit = {"id": p.id, "rrf_score": 0.1, **p.payload}
                text_hits.append(hit)
        except Exception:
            pass

        # Try image collection
        try:
            results, _ = client.scroll(
                collection_name=config.IMAGE_COLLECTION,
                scroll_filter=qmodels.Filter(
                    must=[qmodels.FieldCondition(
                        key="doc_name",
                        match=qmodels.MatchValue(value=candidate)
                    )]
                ),
                limit=top_n,
                with_payload=True,
                with_vectors=False,
            )
            for p in results:
                hit = {"id": p.id, "rrf_score": 0.1, **p.payload}
                image_hits.append(hit)
        except Exception:
            pass

    return text_hits, image_hits

def retrieve(query: str, models: SharedModels) -> dict:
    # Step 0: Direct doc_name lookup — if query mentions an uploaded document,
    # fetch it directly from Qdrant without relying on similarity search
    direct_text, direct_image = _direct_lookup_by_docname(query, models.client)

    # Expand with both company aliases and education terms
    expanded = expand_query(query)
    expanded = _expand_education_query(expanded)

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
    # Prepend direct lookup results (uploaded docs) before CDE validation
    if direct_text:
        fused_text  = direct_text  + fused_text
    if direct_image:
        fused_image = direct_image + fused_image

    cde = get_cde()
    validated = cde.validate(
        query      = query,
        text_hits  = fused_text,
        image_hits = fused_image,
        top_k      = TOP_N,
    )

    # Final deduplication by content (same as other architectures)
    # Priority 1: if query mentions a specific doc_name that exists in results,
    # boost those results to the top (handles uploaded documents)
    query_words = set(query.upper().replace("-","_").replace(" ","_").split("_"))
    query_words |= set(query.upper().split())

    def doc_mentioned_in_query(doc_name: str) -> bool:
        if not doc_name:
            return False
        # Check if significant parts of doc_name appear in query
        parts = [p for p in doc_name.upper().replace("-","_").split("_") if len(p) > 3]
        return sum(1 for p in parts if p in query.upper()) >= 2

    # Boost images whose doc_name is mentioned in query
    boosted_imgs = [h for h in validated["image"] if doc_mentioned_in_query(h.get("doc_name",""))]
    other_imgs   = [h for h in validated["image"] if not doc_mentioned_in_query(h.get("doc_name",""))]
    if boosted_imgs:
        validated["image"] = boosted_imgs + other_imgs

    # Boost text whose doc_name is mentioned in query
    boosted_text = [h for h in validated["text"] if doc_mentioned_in_query(h.get("doc_name",""))]
    other_text   = [h for h in validated["text"] if not doc_mentioned_in_query(h.get("doc_name",""))]
    if boosted_text:
        validated["text"] = boosted_text + other_text

    # Priority 2: company-aware filter as fallback
    company_filter = _extract_company(query)
    if company_filter and not boosted_imgs and validated["image"]:
        same_co = [h for h in validated["image"] if company_filter in h.get("doc_name","").upper()]
        if same_co:
            validated["image"] = same_co

    final_text  = deduplicate(validated["text"],  TOP_N)
    final_image = deduplicate(validated["image"], TOP_N)

    # Smart image filter:
    # If top text results are from specific documents, only keep images
    # from those same documents. Prevents unrelated images from appearing
    # when the document has no relevant images (e.g. text-only PDFs).
    if final_text:
        top_docs = set(h.get("doc_name","") for h in final_text if h.get("doc_name"))
        same_doc_images = [h for h in final_image if h.get("doc_name","") in top_docs]
        # Only filter if we have matching images — otherwise keep all
        if same_doc_images:
            final_image = same_doc_images
        elif final_text:
            # No images from the relevant document — return empty image list
            # User gets text answer without irrelevant images cluttering the UI
            final_image = []

    return {
        "text":  final_text,
        "image": final_image,
        "sub_queries": sub_qs if "sub_qs" in dir() else [],
    }