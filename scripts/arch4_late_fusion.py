"""
arch4_late_fusion.py  —  Architecture 4: Late Fusion Multimodal RAG
---------------------------------------------------------------------
Strategy: Retrieve text and image candidates independently, then merge
using WEIGHTED SCORE COMBINATION (not rank-based like RRF).

  final_score = ALPHA * norm(text_score) + (1 - ALPHA) * norm(image_score)

Both score types are normalised to [0, 1] before combining:
  - Text (cosine similarity): already in [0,1] roughly
  - Image (CLIP cosine):      already in [0,1] roughly
  - BM25 scores:              normalised by dividing by max score in batch

ALPHA controls modality weighting:
  - ALPHA=1.0  → pure text
  - ALPHA=0.0  → pure image
  - ALPHA=0.7  → trust text 70%, images 30% (default for financial QA)
  - ALPHA=0.5  → equal weight

This architecture lets you empirically study which modality contributes
more to retrieval quality across different financial question types.

Strengths : Tunable; shows per-modality contribution; fuses cross-modal evidence
Weaknesses: Requires score normalisation; sensitive to ALPHA choice;
            text and image scores may be on different scales despite normalisation
"""

from base_retriever import SharedModels, deduplicate
import config

TOP_N     = 3
CANDIDATE = 20
ALPHA     = 0.7   # weight for text scores; (1-ALPHA) for image scores


def _normalise(scores: list[float]) -> list[float]:
    """Min-max normalise a list of scores to [0, 1]."""
    if not scores:
        return scores
    mn, mx = min(scores), max(scores)
    if mx == mn:
        return [1.0] * len(scores)
    return [(s - mn) / (mx - mn) for s in scores]


def retrieve(query: str, models: SharedModels, alpha: float = ALPHA) -> dict:
    text_vec = models.encode_text(query)
    clip_vec = models.encode_clip(query)

    # --- Retrieve text candidates ---
    try:
        tr = models.client.query_points(
            collection_name=config.TEXT_COLLECTION,
            query=text_vec, limit=CANDIDATE,
        )
        text_hits = [{"id": p.id, "raw_score": p.score, **p.payload} for p in tr.points]
    except Exception:
        text_hits = []

    # --- Retrieve image candidates ---
    try:
        ir = models.client.query_points(
            collection_name=config.IMAGE_COLLECTION,
            query=clip_vec, limit=CANDIDATE,
        )
        image_hits = [{"id": p.id, "raw_score": p.score, **p.payload} for p in ir.points]
    except Exception:
        image_hits = []

    # --- Normalise scores per modality ---
    text_norm  = _normalise([h["raw_score"] for h in text_hits])
    image_norm = _normalise([h["raw_score"] for h in image_hits])

    for h, s in zip(text_hits,  text_norm):
        h["norm_score"] = s
    for h, s in zip(image_hits, image_norm):
        h["norm_score"] = s

    # --- Build unified pool: assign final_score = alpha*text + (1-alpha)*image ---
    # For text-only candidates: final_score = alpha * norm_score + (1-alpha) * 0
    # For image-only candidates: final_score = alpha * 0 + (1-alpha) * norm_score
    # For cross-modal overlap (same doc_name): average if both modalities found it
    pool = {}

    for h in text_hits:
        doc_name = h.get("doc_name", h["id"])
        pool[doc_name] = {
            "text_score":  h["norm_score"],
            "image_score": 0.0,
            "data": h,
        }

    for h in image_hits:
        doc_name = h.get("doc_name", h["id"])
        if doc_name in pool:
            pool[doc_name]["image_score"] = h["norm_score"]
        else:
            pool[doc_name] = {
                "text_score":  0.0,
                "image_score": h["norm_score"],
                "data": h,
            }

    # Compute final weighted score
    ranked = []
    for doc_name, entry in pool.items():
        final = alpha * entry["text_score"] + (1 - alpha) * entry["image_score"]
        item  = entry["data"].copy()
        item["final_score"]  = final
        item["text_score"]   = entry["text_score"]
        item["image_score"]  = entry["image_score"]
        item["alpha"]        = alpha
        ranked.append(item)

    ranked.sort(key=lambda x: x["final_score"], reverse=True)

    # Split back into text/image for consistent output format
    text_out  = [r for r in ranked if not r.get("image_path")]
    image_out = [r for r in ranked if r.get("image_path")]

    return {
        "text":  deduplicate(text_out,  TOP_N),
        "image": deduplicate(image_out, TOP_N),
        "alpha": alpha,
    }
