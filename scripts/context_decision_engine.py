"""
context_decision_engine.py
---------------------------
Implements the Context Decision Engine (CDE) from your architecture diagram.

The CDE takes retrieved chunks and applies 4 validation stages:
  1. Query Context Relevance Scoring   → trained DistilBERT classifier (label 0/1/2)
  2. Redundancy Detection              → cosine similarity + thresholding
  3. Modality Suitability Check        → rule-based text vs image classifier
  4. Validation Confidence Scoring     → logistic aggregation of above scores

Output: validated chunks with scores, ready for re-ranking.

This replaces the simple top-k cutoff in your current retrieval pipeline
with a quality-aware selection that filters irrelevant or redundant chunks
before they reach the LLM.
"""

import os
import sys
import json
import torch
import numpy as np
import re
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODELS_DIR = Path("models/context_validator")
CONFIG_PATH = MODELS_DIR / "training_config.json"
BEST_MODEL  = MODELS_DIR / "best_model"

# Thresholds
REDUNDANCY_SIM_THRESHOLD = 0.85   # cosine > 0.85 = redundant
MIN_CONFIDENCE_SCORE     = 0.30   # chunks below this are filtered out


# ------------------------------------------------------------------ #
# Modality Suitability Check (rule-based)
# ------------------------------------------------------------------ #

# Keywords that indicate a chunk is better answered by images/tables
TABLE_KEYWORDS = [
    "table", "figure", "chart", "graph", "balance sheet", "income statement",
    "cash flow", "ratio", "margin", "growth", "comparison", "%", "quarter",
    "fiscal year", "fy20", "fy21", "fy22", "revenue", "profit", "loss",
]
TEXT_KEYWORDS = [
    "policy", "strategy", "risk", "management", "notes to", "description",
    "accounting", "method", "approach", "overview", "discussion",
]


def modality_suitability(query: str, chunk: str, chunk_modality: str) -> float:
    """
    Returns a suitability score [0, 1] for whether this modality fits the query.
    Text query about a number → image/table chunk is more suitable.
    Text query about policy → text chunk is more suitable.
    """
    q_lower = query.lower()
    c_lower = chunk.lower()

    # Check if query is asking for numbers/tables
    query_wants_numbers = any(k in q_lower for k in [
        "how much", "what is the", "ratio", "rate", "%", "margin",
        "revenue", "profit", "total", "debt", "assets", "liabilities",
        "growth", "compare", "fiscal", "fy20", "fy21", "fy22",
    ])

    if chunk_modality == "image":
        # Image/table chunks are highly suitable for number queries
        score = 0.9 if query_wants_numbers else 0.5
    else:
        # Text chunks: suitable unless query clearly wants a number from a table
        table_content = any(k in c_lower for k in TABLE_KEYWORDS[:8])
        if query_wants_numbers and table_content:
            score = 0.8   # text chunk that contains table data — still good
        elif query_wants_numbers and not table_content:
            score = 0.5   # text chunk for a numeric query — medium suitability
        else:
            score = 0.85  # text chunk for a descriptive query — good

    return score


# ------------------------------------------------------------------ #
# Redundancy Detection
# ------------------------------------------------------------------ #

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def remove_redundant(
    chunks: list[dict],
    embeddings: list[np.ndarray],
    threshold: float = REDUNDANCY_SIM_THRESHOLD,
) -> list[dict]:
    """
    Greedy deduplication: keep a chunk only if it is not too similar
    to any already-kept chunk.
    """
    kept      = []
    kept_embs = []
    for chunk, emb in zip(chunks, embeddings):
        is_redundant = any(
            cosine_sim(emb, ke) > threshold for ke in kept_embs
        )
        if not is_redundant:
            kept.append(chunk)
            kept_embs.append(emb)
    return kept


# ------------------------------------------------------------------ #
# Context Decision Engine
# ------------------------------------------------------------------ #

class ContextDecisionEngine:
    """
    Implements the full CDE pipeline from your architecture diagram:
      relevance scoring → redundancy detection → modality check → confidence scoring
    """

    def __init__(self, model_path: str = str(BEST_MODEL), device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model_loaded = False

        # Load config
        if (MODELS_DIR / "training_config.json").exists():
            with open(MODELS_DIR / "training_config.json") as f:
                self.config = json.load(f)
        else:
            self.config = {"max_length": 256, "num_labels": 3}

        # Try to load trained model — fall back to rule-based if not available
        model_dir = Path(model_path)
        if model_dir.exists():
            try:
                print(f"  [CDE] Loading validation model from {model_dir}...")
                self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
                self.model     = AutoModelForSequenceClassification.from_pretrained(
                    str(model_dir)
                ).to(self.device)
                self.model.eval()
                self._model_loaded = True
                print("  [CDE] Validation model loaded.")
            except Exception as e:
                print(f"  [CDE] WARNING: Could not load model ({e}) — using rule-based fallback.")
        else:
            print(f"  [CDE] No trained model at {model_dir} — using rule-based scoring.")
            print("  [CDE] Run train_context_validator.py first for best results.")

    def score_relevance(self, query: str, chunks: list[dict]) -> list[float]:
        """
        Returns relevance probability for each chunk using the trained classifier.
        Falls back to keyword overlap if model not loaded.
        """
        if not self._model_loaded:
            return self._rule_based_relevance(query, chunks)

        scores = []
        max_len = self.config.get("max_length", 256)
        for chunk in chunks:
            enc = self.tokenizer(
                query, chunk["text"],
                max_length=max_len,
                truncation="only_second",
                padding="max_length",
                return_tensors="pt",
            )
            with torch.no_grad():
                logits = self.model(
                    input_ids=enc["input_ids"].to(self.device),
                    attention_mask=enc["attention_mask"].to(self.device),
                ).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                # Score = weighted sum: 0*P(0) + 0.5*P(1) + 1.0*P(2)
                score = 0.5 * probs[1] + 1.0 * probs[2]
            scores.append(float(score))
        return scores

    def _rule_based_relevance(self, query: str, chunks: list[dict]) -> list[float]:
        """Fallback: simple token overlap between query and chunk."""
        q_tokens = set(re.findall(r"\w+", query.lower()))
        scores   = []
        for chunk in chunks:
            c_tokens = set(re.findall(r"\w+", chunk["text"].lower()))
            overlap  = len(q_tokens & c_tokens) / max(len(q_tokens), 1)
            scores.append(min(overlap * 2, 1.0))   # scale up, cap at 1
        return scores

    def validate(
        self,
        query:       str,
        text_hits:   list[dict],
        image_hits:  list[dict],
        text_model=None,            # SentenceTransformer for embeddings (optional)
        top_k:       int = 3,
    ) -> dict:
        """
        Full CDE pipeline.
        Returns {"text": [...], "image": [...]} with validation scores attached.
        """
        # Combine all chunks with modality tag
        all_chunks = []
        for h in text_hits:
            all_chunks.append({
                "text":     (h.get("text") or "")[:500],
                "modality": "text",
                "original": h,
            })
        for h in image_hits:
            all_chunks.append({
                "text":     (h.get("text") or "")[:500],
                "modality": "image",
                "original": h,
            })

        if not all_chunks:
            return {"text": [], "image": []}

        # Step 1: Query Context Relevance Scoring
        relevance_scores = self.score_relevance(query, all_chunks)

        # Step 2: Modality Suitability Check
        suitability_scores = [
            modality_suitability(query, c["text"], c["modality"])
            for c in all_chunks
        ]

        # Step 3: Redundancy Detection (using simple text overlap as proxy for embedding)
        # Full version uses sentence embeddings — this is the lightweight version
        kept_indices = set()
        seen_texts   = []
        for i, c in enumerate(all_chunks):
            text = c["text"][:200]
            is_dup = any(
                len(set(text.split()) & set(s.split())) / max(len(set(text.split())), 1) > 0.85
                for s in seen_texts
            )
            if not is_dup:
                kept_indices.add(i)
                seen_texts.append(text)

        # Step 4: Validation Confidence Score = 0.6 * relevance + 0.3 * suitability + 0.1 * not_redundant
        final_scores = []
        for i, (rel, suit) in enumerate(zip(relevance_scores, suitability_scores)):
            redundancy_bonus = 0.1 if i in kept_indices else 0.0
            confidence = 0.6 * rel + 0.3 * suit + redundancy_bonus
            final_scores.append(confidence)

        # Attach scores and filter
        for i, chunk in enumerate(all_chunks):
            chunk["original"]["cde_relevance"]   = round(relevance_scores[i], 4)
            chunk["original"]["cde_suitability"] = round(suitability_scores[i], 4)
            chunk["original"]["cde_confidence"]  = round(final_scores[i], 4)
            chunk["original"]["cde_redundant"]   = (i not in kept_indices)

        # Sort by confidence and split back into modalities
        text_scored  = sorted(
            [c["original"] for c in all_chunks if c["modality"] == "text"],
            key=lambda x: x["cde_confidence"], reverse=True
        )
        image_scored = sorted(
            [c["original"] for c in all_chunks if c["modality"] == "image"],
            key=lambda x: x["cde_confidence"], reverse=True
        )

        # Filter out very low confidence chunks, keep top_k
        text_filtered  = [c for c in text_scored  if c["cde_confidence"] >= MIN_CONFIDENCE_SCORE][:top_k]
        image_filtered = [c for c in image_scored if c["cde_confidence"] >= MIN_CONFIDENCE_SCORE][:top_k]

        # Safety fallback: if filter removed everything, keep top_k anyway
        if not text_filtered  and text_scored:  text_filtered  = text_scored[:top_k]
        if not image_filtered and image_scored: image_filtered = image_scored[:top_k]

        return {
            "text":  text_filtered,
            "image": image_filtered,
        }