"""
rag_pipeline.py
---------------
Complete RAG pipeline: Retrieval → Prompt Building → Generation

Combines your existing hybrid retriever (Architecture 2) with
LM Studio's Qwen2.5-VL for the generation step.

Three generation modes:
  - text_only   : context = retrieved text passages only
  - image_only  : context = retrieved images only (vision model reads tables)
  - multimodal  : context = text + images (recommended)

Usage:
    python rag_pipeline.py
    python rag_pipeline.py --mode text_only
    python rag_pipeline.py --mode image_only
    python rag_pipeline.py --mode multimodal
"""

import os
import sys
import json
import argparse
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from your existing scripts directory
scripts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
sys.path.insert(0, scripts_dir)

from base_retriever import SharedModels
import arch1_naive
import arch2_hybrid_rrf
import arch3_metadata_filter
import arch4_late_fusion
import arch5_query_expansion
import arch6_full_proposed

ARCHITECTURES = {
    1: ("Naive Multimodal RAG",   arch1_naive),
    2: ("Hybrid RAG + RRF",       arch2_hybrid_rrf),
    3: ("Metadata-Filtered RAG",  arch3_metadata_filter),
    4: ("Late Fusion RAG",        arch4_late_fusion),
    5: ("Query Expansion RAG",    arch5_query_expansion),
    6: ("Full Proposed (CDE)",    arch6_full_proposed),
}

from lmstudio_client import health_check, chat
from prompt_builder  import (
    build_text_only_prompt,
    build_image_only_prompt,
    build_multimodal_prompt,
)


def run_pipeline(
    query:    str,
    models:   SharedModels,
    mode:     str  = "multimodal",
    verbose:  bool = True,
) -> dict:
    """
    Full RAG pipeline for a single query.

    Args:
        query   : the user's financial question
        models  : loaded SharedModels instance (retriever)
        mode    : "text_only" | "image_only" | "multimodal"
        verbose : print intermediate steps

    Returns dict with keys:
        query, mode, text_hits, image_hits, messages, answer, latency_s
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"  QUERY: {query}")
        print(f"  MODE : {mode}")
        print("="*60)

    # --- Step 1: Retrieve ---
    t0          = time.time()
    arch_module = ARCHITECTURES[arch_id][1]
    results     = arch_module.retrieve(query, models)
    text_hits   = results.get("text",  [])
    image_hits  = results.get("image", [])
    retr_time   = time.time() - t0

    if verbose:
        print(f"\n  Retrieved: {len(text_hits)} text chunks, {len(image_hits)} images ({retr_time:.2f}s)")
        for h in text_hits:
            print(f"    [TEXT]  {h.get('doc_name')} | {(h.get('text') or '')[:80]}...")
        for h in image_hits:
            print(f"    [IMAGE] {h.get('doc_name')} | {h.get('image_path')}")

    # --- Step 2: Build prompt ---
    if mode == "text_only":
        messages = build_text_only_prompt(query, text_hits)
    elif mode == "image_only":
        messages = build_image_only_prompt(query, image_hits)
    else:  # multimodal (default)
        messages = build_multimodal_prompt(query, text_hits, image_hits)

    if verbose:
        print(f"\n  Prompt built ({mode} mode) — sending to LM Studio...")

    # --- Step 3: Generate ---
    t1      = time.time()
    answer  = chat(messages, temperature=0.1, max_tokens=512)
    gen_time = time.time() - t1

    if verbose:
        print(f"\n  ANSWER ({gen_time:.2f}s):")
        print(f"  {answer}")

    return {
        "query":      query,
        "mode":       mode,
        "text_hits":  [{"doc_name": h.get("doc_name"), "text": (h.get("text") or "")[:200]}
                       for h in text_hits],
        "image_hits": [{"doc_name": h.get("doc_name"), "image_path": h.get("image_path")}
                       for h in image_hits],
        "answer":     answer,
        "latency":    {
            "retrieval_s":  round(retr_time,  3),
            "generation_s": round(gen_time,   3),
            "total_s":      round(retr_time + gen_time, 3),
        },
    }


def run_demo(mode: str = "multimodal"):
    """Run a demo with sample financial questions."""

    print("\nChecking LM Studio connection...")
    if not health_check():
        print(
            "ERROR: Cannot connect to LM Studio or model not loaded.\n"
            "  1. Open LM Studio\n"
            "  2. Load the model: qwen2.5-vl-7b-instruct\n"
            "  3. Start the local server (port 1234)\n"
            "  4. Re-run this script"
        )
        return

    print("LM Studio connected. Loading retrieval models...")
    models = SharedModels()

    # Demo questions — mix of entity-specific and generic
    demo_queries = [
        "What is the Long-term Debt to Total Liabilities for COSTCO in FY2021?",
        "What were the gross profit margins for BESTBUY in fiscal 2023?",
        "Show me the revenue trend for NETFLIX across multiple years.",
    ]

    all_results = []
    for query in demo_queries:
        result = run_pipeline(query, models, mode=mode, verbose=True)
        all_results.append(result)

    # Save results
    out_dir  = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "results"
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"pipeline_demo_{mode}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["text_only", "image_only", "multimodal"],
        default="multimodal",
        help="Generation mode (default: multimodal)"
    )
    args = parser.parse_args()
    run_demo(mode=args.mode)