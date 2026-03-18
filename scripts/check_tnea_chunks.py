"""
check_tnea_chunks.py — diagnostic script
Run: python scripts/check_tnea_chunks.py
"""
import sys, os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

sys.path.insert(0, PROJECT_ROOT)

import config
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)

print("=== TNEA_2025 chunks in Qdrant ===")
results, _ = client.scroll(
    collection_name=config.TEXT_COLLECTION,
    scroll_filter=qmodels.Filter(must=[
        qmodels.FieldCondition(key="doc_name",
                               match=qmodels.MatchValue(value="TNEA_2025"))
    ]),
    limit=100, with_payload=True, with_vectors=False,
)
print(f"Total text chunks: {len(results)}")
for p in results:
    text = p.payload.get("text","")
    page = p.payload.get("page_id","?")
    # Show chunks that contain scholarship-related words
    if any(w in text.lower() for w in ["scholar","sc/sca","st ","sc ","tuition","fee","waiver","matric"]):
        print(f"\n  [PAGE {page}] ID={p.id}")
        print(f"  {text[:300]}")

print("\n=== BM25 test: does 'scholarship SC ST' match TNEA chunks? ===")
from base_retriever import SharedModels, tokenize, build_bm25_index
bm25, records = build_bm25_index(client, config.TEXT_COLLECTION)
query_tokens  = tokenize("scholarship SC ST students")
scores        = bm25.get_scores(query_tokens)

import numpy as np
top10_idx = np.argsort(scores)[::-1][:10]
print("Top 10 BM25 results for 'scholarship SC ST students':")
for i, idx in enumerate(top10_idx, 1):
    r    = records[idx]
    doc  = r["payload"].get("doc_name","?")
    page = r["payload"].get("page_id","?")
    txt  = (r["payload"].get("text",""))[:80]
    print(f"  {i}. score={scores[idx]:.2f} | {doc} p.{page} | {txt}...")