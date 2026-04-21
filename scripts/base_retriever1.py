"""
base_retriever.py
-----------------
Shared infrastructure used by all RAG architectures:
  - Model loading (SentenceTransformer + CLIP)
  - BM25 index building
  - Common tokenizer
  - Qdrant client setup
"""

import os
import re
import sys
import logging
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

logger = logging.getLogger(__name__)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", (text or "").lower())


def build_bm25_index(client: QdrantClient, collection_name: str):
    """Scroll all points and build an in-memory BM25 index."""
    records = []
    seen_ids = set()
    offset = None
    while True:
        result, next_offset = client.scroll(
            collection_name=collection_name,
            limit=256, offset=offset,
            with_payload=True, with_vectors=False,
        )
        for p in result:
            if p.id in seen_ids:
                continue
            seen_ids.add(p.id)
            records.append({
                "id": p.id,
                "payload": p.payload,
                "text_tokens": tokenize(p.payload.get("text") or ""),
            })
        if next_offset is None:
            break
        offset = next_offset
    bm25 = BM25Okapi([r["text_tokens"] for r in records])
    return bm25, records


class SharedModels:
    """
    Loads all models once and passes them to every architecture.
    Avoids reloading on each architecture run.
    """
    def __init__(self):
        logger.info("Connecting to Qdrant...")
        self.client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)

        logger.info("Loading SentenceTransformer...")
        self.text_model = SentenceTransformer(config.TEXT_MODEL, device=config.DEVICE)

        logger.info("Loading CLIP...")
        self.clip_model = CLIPModel.from_pretrained(config.IMAGE_MODEL).to(config.DEVICE)
        self.clip_processor = CLIPProcessor.from_pretrained(config.IMAGE_MODEL)
        self.clip_model.eval()

        logger.info("Building BM25 indexes...")
        self.text_bm25, self.text_records = build_bm25_index(self.client, config.TEXT_COLLECTION)
        self.image_bm25, self.image_records = build_bm25_index(self.client, config.IMAGE_COLLECTION)

        logger.info("All models and indexes ready.")

    def encode_text(self, query: str) -> list[float]:
        return self.text_model.encode(query).tolist()

    def encode_clip(self, query: str) -> list[float]:
        """Encode query with CLIP text encoder, handling both tensor and BaseModelOutputWithPooling."""
        inputs = self.clip_processor(
            text=[query], return_tensors="pt", padding=True
        ).to(config.DEVICE)
        with torch.no_grad():
            outputs = self.clip_model.get_text_features(**inputs)

            # Extract the actual feature tensor (for older transformers versions)
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                feats = outputs.pooler_output
            elif hasattr(outputs, 'last_hidden_state'):
                # Use the [CLS] token (first token) representation
                feats = outputs.last_hidden_state[:, 0, :]
            else:
                feats = outputs  # assume it's already a tensor

            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
        return feats.cpu().numpy()[0].tolist()


def deduplicate(results: list[dict], top_n: int = 3) -> list[dict]:
    """
    Deduplicate by image_path (images) or (doc_name, text[:80]) (text).
    Returns top_n unique results.
    """
    seen_paths = set()
    seen_text_key = set()
    out = []
    for r in results:
        path = r.get("image_path")
        if path:
            if path in seen_paths:
                continue
            seen_paths.add(path)
        else:
            key = (r.get("doc_name", ""), (r.get("text") or "")[:80])
            if key in seen_text_key:
                continue
            seen_text_key.add(key)
        out.append(r)
        if len(out) >= top_n:
            break
    return out