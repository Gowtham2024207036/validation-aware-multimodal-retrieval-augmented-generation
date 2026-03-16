"""
arch1_naive.py  —  Architecture 1: Naive Multimodal RAG
--------------------------------------------------------
Strategy: Pure dense vector search, no BM25, no fusion.
  - Text query  → SentenceTransformer 768-dim → text_collection
  - Text query  → CLIP text encoder 512-dim   → image_collection
  - Results returned independently (no cross-modal merging)

Strengths : Simple, fast, understands semantic meaning
Weaknesses: Ignores exact keywords; wrong company retrieved
            when semantic similarity is high across companies
"""

from base_retriever import SharedModels, deduplicate
import config

TOP_N      = 3
CANDIDATE  = 10


def retrieve(query: str, models: SharedModels) -> dict:
    text_vec  = models.encode_text(query)
    clip_vec  = models.encode_clip(query)

    # --- Text collection ---
    try:
        tr = models.client.query_points(
            collection_name=config.TEXT_COLLECTION,
            query=text_vec, limit=CANDIDATE,
        )
        text_hits = [{"id": p.id, "score": p.score, **p.payload} for p in tr.points]
    except Exception as e:
        print(f"  [Arch1] Text search error: {e}")
        text_hits = []

    # --- Image collection ---
    try:
        ir = models.client.query_points(
            collection_name=config.IMAGE_COLLECTION,
            query=clip_vec, limit=CANDIDATE,
        )
        image_hits = [{"id": p.id, "score": p.score, **p.payload} for p in ir.points]
    except Exception as e:
        print(f"  [Arch1] Image search error: {e}")
        image_hits = []

    return {
        "text":  deduplicate(text_hits,  TOP_N),
        "image": deduplicate(image_hits, TOP_N),
    }
