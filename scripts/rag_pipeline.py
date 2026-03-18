"""
rag_pipeline.py
---------------
Full end-to-end RAG pipeline: Retrieve → Build Prompt → Generate.

Usage:
    # Single interactive query
    python scripts/rag_pipeline.py --arch 2 --mode multimodal

    # Single query from command line
    python scripts/rag_pipeline.py --arch 6 --mode multimodal --query "What is COSTCO revenue FY2021?"

    # Batch evaluation on test_set.jsonl
    python scripts/rag_pipeline.py --arch 2 --mode multimodal --batch --sample 20
    python scripts/rag_pipeline.py --arch 6 --mode multimodal --batch --sample 20
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_retriever import SharedModels
from lmstudio_client import (
    check_connection,
    generate_text_only,
    generate_with_images,
    LM_STUDIO_BASE_URL,
    LM_STUDIO_MODEL,
)
from prompt_builder import (
    SYSTEM_PROMPT,
    build_text_only_prompt,
    build_multimodal_prompt,
    format_answer,
)

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

IMAGES_ROOT   = Path("data/raw")
TEST_SET_PATH = Path("data/raw/test_set.jsonl")
RESULTS_DIR   = Path("data/results/generated_answers")


# ------------------------------------------------------------------ #
# Single query
# ------------------------------------------------------------------ #

def run_single(
    query:    str,
    arch_id:  int,
    mode:     str,
    models:   SharedModels,
    verbose:  bool = True,
) -> dict:
    """Run retrieval + generation for one query. Returns result dict."""
    arch_name, arch_module = ARCHITECTURES[arch_id]

    # Step 1 — Retrieve
    t0     = time.time()
    result = arch_module.retrieve(query, models)
    t_ret  = time.time() - t0

    text_hits  = result.get("text",  [])
    image_hits = result.get("image", [])

    if verbose:
        print(f"\n  [{arch_name}] Retrieved: {len(text_hits)} text, "
              f"{len(image_hits)} images  ({t_ret*1000:.0f}ms)")
        for h in text_hits:
            print(f"    TEXT:  {h.get('doc_name')} | {(h.get('text') or '')[:80]}...")
        for h in image_hits:
            print(f"    IMAGE: {h.get('doc_name')} | {h.get('image_path')}")

    # Step 2 — Build prompt + Generate
    t1 = time.time()

    # Build prompt — always include image captions as text context
    user_text, image_paths = build_multimodal_prompt(query, text_hits, image_hits)

    if mode == "multimodal" and image_paths:
        answer = generate_with_images(
            system_prompt = SYSTEM_PROMPT,
            user_text     = user_text,
            image_paths   = image_paths,
            images_root   = str(IMAGES_ROOT),
            max_tokens    = 1024,
            temperature   = 0.1,
        )
    else:
        # text_only mode OR no images retrieved — always reliable
        answer = generate_text_only(
            system_prompt = SYSTEM_PROMPT,
            user_text     = user_text,
            max_tokens    = 1024,
            temperature   = 0.1,
        )

    # Safety guard — should never happen but prevents None in output
    if answer is None or not isinstance(answer, str):
        answer = generate_text_only(
            system_prompt = SYSTEM_PROMPT,
            user_text     = user_text,
            max_tokens    = 1024,
            temperature   = 0.1,
        )

    t_gen = time.time() - t1

    if verbose:
        print(f"\n  Answer ({t_gen*1000:.0f}ms):\n  {answer}")

    return format_answer(query, answer, text_hits, image_hits, arch_name, mode) | {
        "retrieval_ms":  round(t_ret * 1000),
        "generation_ms": round(t_gen * 1000),
        "total_ms":      round((t_ret + t_gen) * 1000),
    }


# ------------------------------------------------------------------ #
# Batch mode
# ------------------------------------------------------------------ #

def run_batch(arch_id: int, mode: str, models: SharedModels, sample: int = 20):
    """Run pipeline on first `sample` questions from test_set.jsonl."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not TEST_SET_PATH.exists():
        print(f"ERROR: {TEST_SET_PATH} not found.")
        return

    questions = []
    with open(TEST_SET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    questions = questions[:sample]

    arch_name = ARCHITECTURES[arch_id][0]
    print(f"\nBatch run: arch={arch_id} ({arch_name}), mode={mode}, n={len(questions)}")

    outputs = []
    for i, q in enumerate(questions):
        question = q.get("question", "")
        if not question:
            continue
        print(f"\n[{i+1}/{len(questions)}] {question[:80]}")
        result = run_single(question, arch_id, mode, models, verbose=True)
        result["test_id"]        = q.get("test_id")
        result["gold_quote_ids"] = q.get("gold_quote_ids", [])
        result["reference"]      = q.get("reference_answer", "")
        outputs.append(result)

    out_path = RESULTS_DIR / f"arch{arch_id}_{mode}_{len(outputs)}q.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)

    avg_ret = sum(r["retrieval_ms"]  for r in outputs) / max(len(outputs), 1)
    avg_gen = sum(r["generation_ms"] for r in outputs) / max(len(outputs), 1)
    print(f"\nSaved {len(outputs)} answers → {out_path}")
    print(f"Avg retrieval: {avg_ret:.0f}ms | Avg generation: {avg_gen:.0f}ms")


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="End-to-end Multimodal RAG Pipeline")
    parser.add_argument("--arch",   type=int, default=2, choices=[1,2,3,4,5,6])
    parser.add_argument("--mode",   type=str, default="multimodal",
                        choices=["text_only", "multimodal"])
    parser.add_argument("--query",  type=str, default=None,
                        help="Single query string")
    parser.add_argument("--batch",  action="store_true",
                        help="Run batch on test_set.jsonl")
    parser.add_argument("--sample", type=int, default=20,
                        help="Number of questions for batch mode")
    args = parser.parse_args()

    # Check LM Studio
    print(f"Checking LM Studio at {LM_STUDIO_BASE_URL}...")
    if not check_connection():
        print(f"ERROR: Cannot connect to LM Studio.")
        print(f"  1. Open LM Studio")
        print(f"  2. Load model: {LM_STUDIO_MODEL}")
        print(f"  3. Start the local server")
        sys.exit(1)
    print(f"LM Studio connected. Model: {LM_STUDIO_MODEL}")

    # Load retrieval models
    print("\nLoading retrieval models...")
    models = SharedModels()

    arch_name = ARCHITECTURES[args.arch][0]
    print(f"Architecture: {args.arch} — {arch_name}")
    print(f"Mode:         {args.mode}")

    if args.batch:
        run_batch(args.arch, args.mode, models, sample=args.sample)
    else:
        query = args.query or input("\nEnter your question: ").strip()
        if not query:
            print("No query provided.")
            sys.exit(1)
        print(f"\nQuery: {query}")
        run_single(query, args.arch, args.mode, models, verbose=True)


if __name__ == "__main__":
    main()