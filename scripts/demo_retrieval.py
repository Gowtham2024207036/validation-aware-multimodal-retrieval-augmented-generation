"""
Hybrid Retrieval: BM25 (keyword) + Dense Vector (semantic) search
------------------------------------------------------------------
WHY HYBRID?
  - Pure vector search: finds semantically similar docs but ignores
    exact keywords like "COSTCO" or "FY2021" — wrong company retrieved.
  - Pure BM25: finds exact keyword matches but misses paraphrases and
    synonyms.
  - Hybrid (RRF fusion): combines both signals so the correct company
    AND the right financial concept both contribute to the final rank.

ARCHITECTURE:
  - BM25 index: built in-memory from all text/image payloads using
    rank_bm25. Operates on the "text" field stored in Qdrant payloads.
  - Vector index: Qdrant dense search (existing collections).
  - Fusion: Reciprocal Rank Fusion (RRF) — standard, parameter-free
    method to merge ranked lists without needing to tune score weights.
"""

import os
import sys
import math
import logging
from collections import defaultdict

import torch
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logging.basicConfig(level=logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

# RRF constant — 60 is the standard value from the original paper.
# Higher = reduces the impact of rank position differences.
RRF_K = 60

# Fix: Wider candidate pools so BM25 and vector lists overlap more,
# giving RRF more signal to work with. 10 was too narrow — high-BM25
# docs were missing from vector top-10 and got no fusion boost.
VECTOR_CANDIDATE_LIMIT = 50
BM25_CANDIDATE_LIMIT   = 50

# Final top-N to show
TOP_N = 3

# Score below this = likely irrelevant
RELEVANCE_THRESHOLD = 0.25

# Fix: Query expansion for entity-specific financial queries.
# Appending ticker + "annual report" boosts BM25 IDF weight for the
# company name, preventing generic financial terms from dominating.
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
    """
    Appends ticker/alias context when a known company is mentioned.
    This raises the BM25 term weight for the entity so generic
    financial terms don't overshadow the company identifier.
    """
    q_lower    = query.lower()
    expansions = [exp for kw, exp in COMPANY_ALIASES.items() if kw in q_lower]
    return (query + " " + " ".join(expansions)).strip() if expansions else query


# ------------------------------------------------------------------
# BM25 Index Builder
# ------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    """Lowercase, split on whitespace and punctuation."""
    import re
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def build_bm25_index(client: QdrantClient, collection_name: str) -> tuple[BM25Okapi, list[dict]]:
    """
    Scrolls ALL points from a Qdrant collection and builds an in-memory
    BM25 index over their 'text' payload field.

    Returns:
        bm25   : BM25Okapi index
        records: list of payload dicts in the same order as BM25 corpus
    """
    records  = []
    seen_ids = set()   # Fix: deduplicate — scroll pagination can return same point twice
    offset   = None

    print(f"  Building BM25 index for '{collection_name}'...")
    while True:
        result, next_offset = client.scroll(
            collection_name=collection_name,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in result:
            if point.id in seen_ids:
                continue          # skip duplicate
            seen_ids.add(point.id)
            records.append({
                "id":          point.id,
                "payload":     point.payload,
                "text_tokens": tokenize(point.payload.get("text") or ""),
            })
        if next_offset is None:
            break
        offset = next_offset

    corpus = [r["text_tokens"] for r in records]
    bm25   = BM25Okapi(corpus)
    print(f"  BM25 index built: {len(records)} documents.")
    return bm25, records


# ------------------------------------------------------------------
# Reciprocal Rank Fusion
# ------------------------------------------------------------------

def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    k: int = RRF_K,
) -> list[dict]:
    """
    Merges multiple ranked lists using RRF.
    Each item in a ranked list must have an 'id' field.
    Returns a unified ranked list sorted by descending RRF score,
    with each item carrying all its original payload fields.
    """
    rrf_scores = defaultdict(float)
    id_to_item = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            doc_id = item["id"]
            rrf_scores[doc_id] += 1.0 / (k + rank)
            if doc_id not in id_to_item:
                id_to_item[doc_id] = item

    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
    results = []
    for doc_id in sorted_ids:
        item = id_to_item[doc_id].copy()
        item["rrf_score"] = rrf_scores[doc_id]
        results.append(item)

    return results


# ------------------------------------------------------------------
# Hybrid Search: Text Collection
# ------------------------------------------------------------------

def hybrid_search_text(
    client:     QdrantClient,
    text_model: SentenceTransformer,
    bm25:       BM25Okapi,
    records:    list[dict],
    query:      str,
    top_n:      int = TOP_N,
) -> list[dict]:
    """
    Combines BM25 keyword ranking + SentenceTransformer vector ranking
    for the text collection, fused via RRF.
    """
    # Fix: expand query with company alias/ticker before encoding
    expanded_query = expand_query(query)
    if expanded_query != query:
        print(f"  Query expanded: {expanded_query}")

    # --- BM25 candidates ---
    query_tokens = tokenize(expanded_query)
    bm25_scores  = bm25.get_scores(query_tokens)

    # Get top BM25_CANDIDATE_LIMIT indices sorted by score descending
    top_bm25_idx = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True,
    )[:BM25_CANDIDATE_LIMIT]

    bm25_ranked = []
    for idx in top_bm25_idx:
        rec = records[idx]
        bm25_ranked.append({
            "id":         rec["id"],
            "bm25_score": float(bm25_scores[idx]),
            **rec["payload"],
        })

    # --- Vector candidates (use expanded query for better entity matching) ---
    query_vector = text_model.encode(expanded_query).tolist()
    try:
        result = client.query_points(
            collection_name=config.TEXT_COLLECTION,
            query=query_vector,
            limit=VECTOR_CANDIDATE_LIMIT,
        )
        vector_ranked = [
            {"id": p.id, "vector_score": p.score, **p.payload}
            for p in result.points
        ]
    except Exception as e:
        print(f"  Vector search error (text): {e}")
        vector_ranked = []

    # --- RRF Fusion ---
    fused = reciprocal_rank_fusion([bm25_ranked, vector_ranked])
    return fused[:top_n]


# ------------------------------------------------------------------
# Hybrid Search: Image Collection
# ------------------------------------------------------------------

def hybrid_search_images(
    client:          QdrantClient,
    clip_model:      CLIPModel,
    clip_processor:  CLIPProcessor,
    bm25:            BM25Okapi,
    records:         list[dict],
    query:           str,
    top_n:           int = TOP_N,
) -> list[dict]:
    """
    Combines BM25 keyword ranking (over image captions/descriptions) +
    CLIP text-to-image vector ranking, fused via RRF.
    """
    # Fix: expand query with company alias/ticker
    expanded_query = expand_query(query)

    # --- BM25 candidates (over image caption text) ---
    query_tokens = tokenize(expanded_query)
    bm25_scores  = bm25.get_scores(query_tokens)

    top_bm25_idx = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True,
    )[:BM25_CANDIDATE_LIMIT]

    bm25_ranked = []
    for idx in top_bm25_idx:
        rec = records[idx]
        bm25_ranked.append({
            "id":         rec["id"],
            "bm25_score": float(bm25_scores[idx]),
            **rec["payload"],
        })

    # --- CLIP vector candidates (use expanded query) ---
    inputs = clip_processor(text=[expanded_query], return_tensors="pt", padding=True).to(config.DEVICE)
    with torch.no_grad():
        text_features = clip_model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    query_vector = text_features.cpu().numpy().tolist()[0]

    try:
        result = client.query_points(
            collection_name=config.IMAGE_COLLECTION,
            query=query_vector,
            limit=VECTOR_CANDIDATE_LIMIT,
        )
        vector_ranked = [
            {"id": p.id, "vector_score": p.score, **p.payload}
            for p in result.points
        ]
    except Exception as e:
        print(f"  Vector search error (image): {e}")
        vector_ranked = []

    # --- RRF Fusion ---
    fused = reciprocal_rank_fusion([bm25_ranked, vector_ranked])
    return fused[:top_n]


# ------------------------------------------------------------------
# Display
# ------------------------------------------------------------------

def print_results(hits: list[dict], label: str):
    """Pretty-print hybrid results with both individual and fused scores."""
    print(f"\n  TOP {len(hits)} {label} RESULTS (Hybrid BM25 + Vector):")
    print("  " + "-" * 60)

    if not hits:
        print("  No results returned.")
        return

    for i, hit in enumerate(hits, 1):
        rrf_score    = hit.get("rrf_score", 0)
        vector_score = hit.get("vector_score")
        bm25_score   = hit.get("bm25_score")
        doc_name     = hit.get("doc_name", "unknown")
        image_path   = hit.get("image_path")
        text         = (hit.get("text") or "")
        desc         = text[:120] + ("..." if len(text) > 120 else "")

        score_parts = [f"RRF: {rrf_score:.4f}"]
        if vector_score is not None:
            score_parts.append(f"Vector: {vector_score:.4f}")
        if bm25_score is not None:
            score_parts.append(f"BM25: {bm25_score:.2f}")

        flag = ""
        if vector_score is not None and vector_score < RELEVANCE_THRESHOLD:
            flag = "  [low vector score]"

        print(f"  Rank {i} | {' | '.join(score_parts)}{flag}")
        print(f"  Doc  : {doc_name}")
        if image_path:
            print(f"  Path : {image_path}")
        print(f"  Text : {desc}")
        print("  " + "-" * 60)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def run_hybrid_retrieval():
    print("=" * 62)
    print("  HYBRID RETRIEVAL TEST  (BM25 + Vector → RRF Fusion)")
    print("=" * 62)

    # Connect
    try:
        client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
        print(f"\nConnected to Qdrant at {config.QDRANT_HOST}:{config.QDRANT_PORT}")
    except Exception as e:
        print(f"FATAL: Cannot connect to Qdrant — is it running?\n  {e}")
        return

    # Verify collections
    print("\nVerifying collections...")
    for name in [config.TEXT_COLLECTION, config.IMAGE_COLLECTION]:
        try:
            count = client.get_collection(name).points_count
            print(f"  '{name}': {count:,} points")
        except Exception as e:
            print(f"  ERROR accessing '{name}': {e}")
            return

    # Build BM25 indexes (one per collection)
    print("\nBuilding BM25 indexes...")
    text_bm25,  text_records  = build_bm25_index(client, config.TEXT_COLLECTION)
    image_bm25, image_records = build_bm25_index(client, config.IMAGE_COLLECTION)

    # Load models
    print(f"\nLoading text model ({config.TEXT_MODEL}) on {config.DEVICE.upper()}...")
    text_model = SentenceTransformer(config.TEXT_MODEL, device=config.DEVICE)

    print(f"Loading CLIP model ({config.IMAGE_MODEL}) on {config.DEVICE.upper()}...")
    clip_model     = CLIPModel.from_pretrained(config.IMAGE_MODEL).to(config.DEVICE)
    clip_processor = CLIPProcessor.from_pretrained(config.IMAGE_MODEL)
    clip_model.eval()

    # Test queries
    test_queries = [
        "What is the Long-term Debt to Total Liabilities for COSTCO in FY2021?",
        "Show me a table comparing revenue across fiscal years.",
        "What are the gross profit margins for Amazon in 2022?",
    ]

    for query in test_queries:
        print(f"\n{'=' * 62}")
        print(f"  QUERY: {query}")
        print("=" * 62)

        text_hits = hybrid_search_text(
            client, text_model, text_bm25, text_records, query
        )
        print_results(text_hits, "TEXT")

        image_hits = hybrid_search_images(
            client, clip_model, clip_processor, image_bm25, image_records, query
        )
        print_results(image_hits, "IMAGE")

    print("\nHybrid retrieval test complete.")


if __name__ == "__main__":
    run_hybrid_retrieval()