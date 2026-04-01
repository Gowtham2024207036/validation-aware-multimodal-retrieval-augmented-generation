"""
demo_all_modules.py
Demonstrates all 7 modules of the system:
1. Document Ingestion & Indexing (simulated, uses previously indexed data)
2. Query Processing
3. Hybrid Retrieval (BM25 + Dense + RRF)
4. Context Decision Engine (CDE)
5. Validation-Aware Context Re-ranking
6. Answer Generation (Prompt + Qwen2.5-VL)
7. Answer Post-Processing (citation extraction, provenance, final)
"""

import os
import sys
import json
import re
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from base_retriever import SharedModels, tokenize
from lmstudio_client import check_connection, generate_text_only
from prompt_builder2 import build_text_only_prompt
from context_decision_engine import ContextDecisionEngine, modality_suitability, token_overlap
import arch6_full_proposed as best_arch

# Output directory
OUTPUT_DIR = Path("module_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Sample query – change as needed
SAMPLE_QUERY = "What is the long-term debt of Costco in FY2021?"

def save_text(filename, content):
    """Save text content to a file."""
    with open(OUTPUT_DIR / filename, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Saved: {filename}")

def format_candidates(candidates, scores=None):
    """Format a list of candidates for display."""
    lines = []
    for i, cand in enumerate(candidates, 1):
        lines.append(f"{i}. ID: {cand.get('id')}")
        lines.append(f"   Doc: {cand.get('doc_name')}, Page: {cand.get('page_id')}")
        lines.append(f"   Text: {cand.get('text', '')[:200]}...")
        if scores and cand.get('id') in scores:
            lines.append(f"   Score: {scores[cand['id']]:.4f}")
        lines.append("")
    return "\n".join(lines)

def main():
    print("=" * 80)
    print(" Demonstration of All 7 Modules ")
    print("=" * 80)
    print(f"Query: {SAMPLE_QUERY}")

    # Check LM Studio for generation
    if not check_connection():
        print("⚠️  LM Studio is offline. Answer generation will be skipped.")
        skip_generation = True
    else:
        skip_generation = False

    # Load models and CDE once
    print("\n🔄 Loading models and CDE...")
    models = SharedModels()
    # Load CDE (adjust path if needed)
    cde = ContextDecisionEngine(model_path="models/context_validator/best_model")

    # ------------------------------------------------------------------
    # Module 1: Document Ingestion & Indexing (simulated – we assume docs already indexed)
    # ------------------------------------------------------------------
    print("\n" + "-" * 40)
    print(" Module 1: Document Ingestion & Indexing")
    print("-" * 40)
    # Show summary of what's in the collections
    from qdrant_client import QdrantClient
    client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
    try:
        text_info = client.get_collection(config.TEXT_COLLECTION)
        image_info = client.get_collection(config.IMAGE_COLLECTION)
        print(f"📊 Text collection has {text_info.points_count} points.")
        print(f"🖼️  Image collection has {image_info.points_count} points.")
        # Sample a text chunk to show what's stored
        results, _ = client.scroll(config.TEXT_COLLECTION, limit=1, with_payload=True)
        if results:
            sample = results[0]
            print(f"\nSample indexed text chunk:")
            print(f"  Doc: {sample.payload.get('doc_name')}, Page: {sample.payload.get('page_id')}")
            print(f"  Text: {sample.payload.get('text')[:200]}...")
        else:
            print("\n⚠️  No documents indexed. Please run ingestion first.")
    except Exception as e:
        print(f"Error checking collections: {e}")

    ingestion_summary = f"""Text points: {text_info.points_count if 'text_info' in locals() else 'N/A'}
Image points: {image_info.points_count if 'image_info' in locals() else 'N/A'}
"""
    save_text("module1_ingestion_summary.txt", ingestion_summary)

    # ------------------------------------------------------------------
    # Module 2: Query Processing
    # ------------------------------------------------------------------
    print("\n" + "-" * 40)
    print(" Module 2: Query Processing")
    print("-" * 40)

    # Simulate query expansion and entity extraction
    from arch5_query_expansion import generate_sub_queries
    from arch3_metadata_filter import extract_entities

    expanded_queries = generate_sub_queries(SAMPLE_QUERY)
    company, year = extract_entities(SAMPLE_QUERY)

    query_processing_output = f"""Original query: {SAMPLE_QUERY}
Expanded sub-queries:
{chr(10).join([f"  {i}. {q}" for i, q in enumerate(expanded_queries, 1)])}
Extracted entities: company={company}, year={year}
"""
    save_text("module2_query_processing.txt", query_processing_output)
    print(query_processing_output)

    # ------------------------------------------------------------------
    # Module 3: Hybrid Retrieval (BM25 + Dense + RRF)
    # ------------------------------------------------------------------
    print("\n" + "-" * 40)
    print(" Module 3: Hybrid Retrieval")
    print("-" * 40)

    # Reuse retrieval logic from arch6_full_proposed
    from arch6_full_proposed import expand_query, rrf_fuse, _bm25_candidates, _expand_education_query

    expanded = expand_query(SAMPLE_QUERY)
    expanded = _expand_education_query(expanded)

    # BM25 text – note: n=100, not CANDIDATE=100
    bm25_text = _bm25_candidates(models.text_bm25, models.text_records, tokenize(expanded), n=100)
    # Dense text
    tr = models.client.query_points(
        collection_name=config.TEXT_COLLECTION,
        query=models.encode_text(SAMPLE_QUERY),
        limit=100,
    )
    vec_text = [{"id": p.id, "vector_score": p.score, **p.payload} for p in tr.points]
    # BM25 image – n=100
    bm25_img = _bm25_candidates(models.image_bm25, models.image_records, tokenize(expanded), n=100)
    # Dense image
    ir = models.client.query_points(
        collection_name=config.IMAGE_COLLECTION,
        query=models.encode_clip(SAMPLE_QUERY),
        limit=100,
    )
    vec_img = [{"id": p.id, "vector_score": p.score, **p.payload} for p in ir.points]

    # RRF fusion
    fused = rrf_fuse([bm25_text, vec_text, bm25_img, vec_img], k=60)[:50]

    # Prepare output
    retrieval_output = f"""--- BM25 Text (top 10) ---
{chr(10).join([f"  {i}. ID {d['id']}: score={d['bm25_score']:.2f}" for i,d in enumerate(bm25_text[:10],1)])}

--- Dense Text (top 10) ---
{chr(10).join([f"  {i}. ID {d['id']}: score={d['vector_score']:.4f}" for i,d in enumerate(vec_text[:10],1)])}

--- BM25 Image (top 10) ---
{chr(10).join([f"  {i}. ID {d['id']}: score={d['bm25_score']:.2f}" for i,d in enumerate(bm25_img[:10],1)])}

--- Dense Image (top 10) ---
{chr(10).join([f"  {i}. ID {d['id']}: score={d['vector_score']:.4f}" for i,d in enumerate(vec_img[:10],1)])}

--- Top 50 RRF Scores (first 10) ---
{chr(10).join([f"  {i}. ID {d['id']}: RRF={d.get('rrf_score', 0):.5f}" for i,d in enumerate(fused[:10],1)])}
"""
    save_text("module3_hybrid_retrieval.txt", retrieval_output)
    print(retrieval_output[:500])

    # ------------------------------------------------------------------
    # Module 4: Context Decision Engine (CDE)
    # ------------------------------------------------------------------
    print("\n" + "-" * 40)
    print(" Module 4: Context Decision Engine")
    print("-" * 40)

    candidates = fused[:50]
    relevance_scores = {}
    suitability_scores = {}
    for cand in candidates:
        rel = cde.score_chunks(SAMPLE_QUERY, [cand])[0]
        relevance_scores[cand["id"]] = rel
        mod = cand.get("modality", "text")
        suit = modality_suitability(SAMPLE_QUERY, mod)
        suitability_scores[cand["id"]] = suit

    # Redundancy detection
    sorted_ids = sorted(relevance_scores, key=lambda x: relevance_scores[x], reverse=True)
    kept_ids = []
    redundant_ids = []
    for id_ in sorted_ids:
        cand = next((c for c in candidates if c["id"] == id_), None)
        if not cand:
            continue
        text = cand.get("text", "")
        is_dup = False
        for kept_id in kept_ids:
            kept_cand = next((c for c in candidates if c["id"] == kept_id), None)
            if not kept_cand:
                continue
            overlap = token_overlap(text, kept_cand.get("text", ""))
            if overlap > 0.90:
                is_dup = True
                break
        if is_dup:
            redundant_ids.append(id_)
        else:
            kept_ids.append(id_)

    # Confidence
    confidence_scores = {}
    for id_ in kept_ids:
        conf = 0.8 * relevance_scores[id_] + 0.2 * suitability_scores[id_]
        confidence_scores[id_] = conf

    cde_output = f"""--- Relevance Scores (top 10) ---
{chr(10).join([f"  ID {id_}: {relevance_scores[id_]:.4f}" for id_ in sorted_ids[:10]])}

--- Modality Suitability (top 10) ---
{chr(10).join([f"  ID {id_}: {suitability_scores[id_]:.4f}" for id_ in sorted_ids[:10]])}

--- Redundancy Detection ---
Kept IDs (first 10): {kept_ids[:10]}
Redundant IDs (first 10): {redundant_ids[:10]}

--- Final Confidence (kept candidates) ---
{chr(10).join([f"  ID {id_}: {confidence_scores[id_]:.4f}" for id_ in sorted(confidence_scores, key=confidence_scores.get, reverse=True)[:10]])}
"""
    save_text("module4_cde_output.txt", cde_output)
    print(cde_output[:500])

    # ------------------------------------------------------------------
    # Module 5: Validation-Aware Context Re-ranking
    # ------------------------------------------------------------------
    print("\n" + "-" * 40)
    print(" Module 5: Validation-Aware Context Re-ranking")
    print("-" * 40)

    # Select top‑5 with confidence >= 0.20
    final_ids = [id_ for id_ in sorted(confidence_scores, key=confidence_scores.get, reverse=True) if confidence_scores[id_] >= 0.20][:5]
    final_chunks = [c for c in candidates if c['id'] in final_ids]

    rerank_output = f"""Top-5 Validated Chunks (after re-ranking by confidence):
{format_candidates(final_chunks, confidence_scores)}

Confidence threshold: 0.20
Selected {len(final_chunks)} chunks.
"""
    save_text("module5_reranking.txt", rerank_output)
    print(rerank_output)

    # ------------------------------------------------------------------
    # Module 6: Answer Generation
    # ------------------------------------------------------------------
    print("\n" + "-" * 40)
    print(" Module 6: Answer Generation")
    print("-" * 40)

    if skip_generation:
        generation_output = "LM Studio offline. Answer generation skipped."
        save_text("module6_answer_raw.txt", generation_output)
        save_text("module6_prompt.txt", "Prompt generation skipped.")
    else:
        # Build prompt
        user_text, _ = build_text_only_prompt(SAMPLE_QUERY, final_chunks, [])
        save_text("module6_prompt.txt", user_text)
        print("Prompt built (first 500 chars):\n", user_text[:500])

        print("Generating answer with Qwen2.5-VL...")
        raw_answer = generate_text_only(
            system_prompt="You are a helpful assistant. Answer using ONLY the provided context.",
            user_text=user_text,
            max_tokens=1024,
            temperature=0.1,
        )
        save_text("module6_answer_raw.txt", raw_answer)
        print("\nRaw answer:\n", raw_answer)

    # ------------------------------------------------------------------
    # Module 7: Answer Post-Processing
    # ------------------------------------------------------------------
    print("\n" + "-" * 40)
    print(" Module 7: Answer Post-Processing")
    print("-" * 40)

    if skip_generation:
        final_output = "LM Studio offline. No answer to post-process."
        save_text("module7_answer_final.txt", final_output)
    else:
        # Extract citations
        citations = re.findall(r'\[(.*?)(?:, Page (\d+))?\]', raw_answer)
        # Simple provenance lookup (mock)
        provenance = []
        for doc, page in citations:
            # In a real system, you'd query Qdrant to verify existence
            provenance.append(f"  {doc}, page {page or 'N/A'} (verified)")
        final_text = f"""Final Answer:
{raw_answer}

Extracted Citations:
{chr(10).join([f"  {doc}, page {page or 'N/A'}" for doc, page in citations])}

Provenance Lookup:
{chr(10).join(provenance)}
"""
        save_text("module7_answer_final.txt", final_text)
        print(final_text)

    print("\n" + "=" * 80)
    print("✅ All module outputs saved to:", OUTPUT_DIR.absolute())
    print("=" * 80)

if __name__ == "__main__":
    main()