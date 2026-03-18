"""
context_decision_engine.py
---------------------------
Context Decision Engine (CDE) — implements the validation layer from
your proposed architecture diagram.

4-stage pipeline:
  1. Query Context Relevance Scoring   — DistilBERT classifier (label 0/1/2)
  2. Redundancy Detection              — cosine similarity thresholding
  3. Modality Suitability Check        — rule-based text vs image routing
  4. Validation Confidence Scoring     — weighted combination of above

Key fix: the CDE re-ranks by relevance score (predicted label-2 probability)
rather than a composite score that can bury gold chunks. The trained model
directly predicts whether each chunk is a gold citation.
"""

import os
import sys
import re
import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODELS_DIR  = Path("models/context_validator")
BEST_MODEL  = MODELS_DIR / "best_model"
CONFIG_PATH = MODELS_DIR / "training_config.json"

# Minimum CDE confidence to keep a chunk (applied only when model is loaded)
MIN_CONFIDENCE = 0.20   # lowered — prevents over-filtering with trained model
# Redundancy threshold — cosine similarity above this = duplicate
REDUNDANCY_THRESHOLD = 0.90


# ------------------------------------------------------------------ #
# Modality suitability (rule-based)
# ------------------------------------------------------------------ #

def modality_suitability(query: str, modality: str) -> float:
    """
    Returns suitability score [0,1] for this modality given the query.
    Numeric/table queries → images more suitable.
    Policy/description queries → text more suitable.
    """
    q = query.lower()
    wants_number = any(k in q for k in [
        "how much", "what is the", "ratio", "rate", "%", "margin",
        "revenue", "profit", "total", "debt", "assets", "liabilities",
        "growth", "compare", "fiscal", "fy20", "fy21", "fy22", "fy23",
        "quarter", "annual", "calculate", "compute", "figure",
    ])
    if modality == "image":
        return 0.85 if wants_number else 0.55
    else:  # text
        return 0.70 if wants_number else 0.85


# ------------------------------------------------------------------ #
# Redundancy detection (token overlap — no embedding needed)
# ------------------------------------------------------------------ #

def token_overlap(a: str, b: str) -> float:
    ta = set(re.findall(r"\w+", a.lower()))
    tb = set(re.findall(r"\w+", b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta), len(tb))


def remove_redundant(chunks: list[dict], threshold: float = REDUNDANCY_THRESHOLD) -> list[dict]:
    """Greedy deduplication by token overlap."""
    kept       = []
    kept_texts = []
    for c in chunks:
        text = (c.get("text") or "")[:300]
        is_dup = any(token_overlap(text, kt) > threshold for kt in kept_texts)
        if not is_dup:
            kept.append(c)
            kept_texts.append(text)
    return kept


# ------------------------------------------------------------------ #
# Context Decision Engine
# ------------------------------------------------------------------ #

class ContextDecisionEngine:
    """
    Validates and re-ranks retrieved chunks using the trained DistilBERT
    context validation model (F1 = 0.9274).

    The model predicts P(label=2) for each (query, chunk) pair.
    label-2 = "relevant" (cited in gold answer).
    Chunks are re-ranked by this probability, with modality suitability
    as a small secondary signal.
    """

    def __init__(self, model_path: str = str(BEST_MODEL)):
        self.device        = "cuda" if torch.cuda.is_available() else "cpu"
        self._model_loaded = False
        self.max_length    = 256

        # Load config
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH) as f:
                cfg = json.load(f)
            self.max_length = cfg.get("max_length", 256)

        # Load trained model
        model_dir = Path(model_path)
        if model_dir.exists():
            try:
                print(f"  [CDE] Loading trained DistilBERT from {model_dir}...")
                self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
                self.model     = AutoModelForSequenceClassification.from_pretrained(
                    str(model_dir)
                ).to(self.device)
                self.model.eval()
                self._model_loaded = True
                print(f"  [CDE] Model loaded on {self.device.upper()}. F1=0.9274")
            except Exception as e:
                print(f"  [CDE] WARNING: Model load failed ({e}) — using rule-based fallback.")
        else:
            print(f"  [CDE] No model at {model_dir} — using rule-based fallback.")
            print("  [CDE] Run train_context_validator.py to train the model.")

    # ── Relevance scoring ────────────────────────────────────────────

    def _score_batch(self, query: str, texts: list[str]) -> list[float]:
        """
        Score a batch of texts with the trained model.
        Returns P(label=2) — probability of being a gold cited chunk.
        """
        probs = []
        for text in texts:
            enc = self.tokenizer(
                query, text,
                max_length=self.max_length,
                truncation="only_second",
                padding="max_length",
                return_tensors="pt",
            )
            with torch.no_grad():
                logits = self.model(
                    input_ids=enc["input_ids"].to(self.device),
                    attention_mask=enc["attention_mask"].to(self.device),
                ).logits
                # Softmax → P(label=2) = "relevant"
                p = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                # p[0]=irrelevant, p[1]=partial, p[2]=relevant
                # If model only has 2 outputs (no label-0 in training):
                if len(p) == 3:
                    probs.append(float(p[2]))
                else:
                    # 2-class model: p[0]=partial, p[1]=relevant
                    probs.append(float(p[-1]))
        return probs

    def _rule_based_relevance(self, query: str, texts: list[str]) -> list[float]:
        """Fallback: Jaccard token overlap as relevance proxy."""
        q_tokens = set(re.findall(r"\w+", query.lower()))
        scores   = []
        for text in texts:
            t_tokens = set(re.findall(r"\w+", text.lower()))
            overlap  = len(q_tokens & t_tokens) / max(len(q_tokens), 1)
            scores.append(min(overlap * 3, 1.0))
        return scores

    def score_chunks(self, query: str, chunks: list[dict]) -> list[float]:
        """Score all chunks — returns relevance score per chunk."""
        texts = [(c.get("text") or "")[:400] for c in chunks]
        if self._model_loaded:
            return self._score_batch(query, texts)
        return self._rule_based_relevance(query, texts)

    # ── Main validate method ─────────────────────────────────────────

    def validate(
        self,
        query:       str,
        text_hits:   list[dict],
        image_hits:  list[dict],
        top_k:       int = 3,
    ) -> dict:
        """
        Full CDE pipeline:
          1. Score all chunks with DistilBERT → P(relevant)
          2. Compute modality suitability
          3. Remove redundant chunks
          4. Re-rank by: 0.80 * relevance + 0.20 * suitability
          5. Return top_k per modality

        The heavy weight on relevance (0.80) ensures the trained model
        drives the ranking. Suitability is a small tie-breaker.
        """
        if not text_hits and not image_hits:
            return {"text": [], "image": []}

        # Combine all chunks for batch scoring
        all_chunks    = []
        all_modalities= []
        for h in text_hits:
            all_chunks.append(h)
            all_modalities.append("text")
        for h in image_hits:
            all_chunks.append(h)
            all_modalities.append("image")

        # Stage 1: Relevance scoring
        relevance_scores = self.score_chunks(query, all_chunks)

        # Stage 2: Modality suitability
        suitability_scores = [
            modality_suitability(query, m)
            for m in all_modalities
        ]

        # Stage 3: Compute final CDE confidence score
        for i, (chunk, rel, suit) in enumerate(
            zip(all_chunks, relevance_scores, suitability_scores)
        ):
            cde_score = 0.80 * rel + 0.20 * suit
            chunk["cde_relevance"]   = round(rel,  4)
            chunk["cde_suitability"] = round(suit, 4)
            chunk["cde_confidence"]  = round(cde_score, 4)

        # Stage 4: Split back by modality, remove redundancy, sort, take top_k
        text_scored  = sorted(
            [c for c, m in zip(all_chunks, all_modalities) if m == "text"],
            key=lambda x: x["cde_confidence"], reverse=True
        )
        image_scored = sorted(
            [c for c, m in zip(all_chunks, all_modalities) if m == "image"],
            key=lambda x: x["cde_confidence"], reverse=True
        )

        # Remove redundant chunks (after sorting — keeps highest scored)
        text_deduped  = remove_redundant(text_scored)
        image_deduped = remove_redundant(image_scored)

        # Filter by minimum confidence
        text_filtered  = [c for c in text_deduped  if c["cde_confidence"] >= MIN_CONFIDENCE]
        image_filtered = [c for c in image_deduped if c["cde_confidence"] >= MIN_CONFIDENCE]

        # Safety fallback — never return empty if input was non-empty
        if not text_filtered  and text_scored:  text_filtered  = text_scored
        if not image_filtered and image_scored: image_filtered = image_scored

        return {
            "text":  text_filtered[:top_k],
            "image": image_filtered[:top_k],
        }