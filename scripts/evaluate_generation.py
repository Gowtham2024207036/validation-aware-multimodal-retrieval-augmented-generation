"""
evaluate_with_generation.py
----------------------------
Extends the retrieval evaluation to include generation quality.

For each of the 5 architectures:
  1. Retrieve top-3 text + image results
  2. Build multimodal prompt
  3. Send to Qwen2.5-VL via LM Studio
  4. Score the answer:
     - Retrieval metrics: Hit@3, Recall@3, MRR, NDCG@3 (same as before)
     - Generation metrics:
         * answer_has_number : does the answer contain a specific figure?
         * answer_cites_doc  : does the answer mention a document name?
         * answer_not_found  : did the model admit it couldn't find the answer?
         * answer_length     : word count (proxy for informativeness)

Usage:
    python evaluate_with_generation.py --sample 20 --mode multimodal
    python evaluate_with_generation.py --sample 50 --arch 2 3
"""

import os
import sys
import json
import re
import math
import time
import argparse
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

scripts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
sys.path.insert(0, scripts_dir)

from base_retriever import SharedModels
import arch1_naive
import arch2_hybrid_rrf
import arch3_metadata_filter
import arch4_late_fusion
import arch5_query_expansion

from lmstudio_client import health_check, chat
from prompt_builder  import build_multimodal_prompt, build_text_only_prompt

ARCHITECTURES = {
    1: ("Naive Multimodal RAG",       arch1_naive),
    2: ("Hybrid RAG + RRF",           arch2_hybrid_rrf),
    3: ("Metadata-Filtered RAG",      arch3_metadata_filter),
    4: ("Late Fusion RAG",            arch4_late_fusion),
    5: ("Query Expansion RAG",        arch5_query_expansion),
}

TEST_SET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "raw", "test_set.jsonl"
)
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "results"
)

NUMBER_RE = re.compile(r"\$?[\d,]+\.?\d*\s*(%|billion|million|thousand)?", re.IGNORECASE)


# ------------------------------------------------------------------
# Retrieval metrics (same as evaluate.py)
# ------------------------------------------------------------------

def extract_quote_ids(hits):
    return [str(h.get("quote_id")) for h in hits if h.get("quote_id")]

def hit_rate(ret, gold):      return 1.0 if any(r in gold for r in ret) else 0.0
def recall_at_k(ret, gold):   return sum(1 for r in ret if r in gold) / len(gold) if gold else 0.0
def precision_at_k(ret, gold):return sum(1 for r in ret if r in gold) / len(ret)  if ret  else 0.0
def mrr(ret, gold):
    for i, r in enumerate(ret, 1):
        if r in gold: return 1.0 / i
    return 0.0
def ndcg(ret, gold, k=3):
    dcg  = sum(1/math.log2(i+1) for i,r in enumerate(ret[:k],1) if r in gold)
    idcg = sum(1/math.log2(i+1) for i in range(1, min(len(gold),k)+1))
    return dcg/idcg if idcg else 0.0


# ------------------------------------------------------------------
# Generation quality heuristics
# ------------------------------------------------------------------

def score_answer(answer: str, gold_doc_names: list[str]) -> dict:
    """
    Lightweight generation quality scoring — no LLM judge needed.
    Uses heuristics appropriate for financial QA.
    """
    answer_lower = answer.lower()
    return {
        # Did the model find a specific number/percentage?
        "has_number":    1 if NUMBER_RE.search(answer) else 0,

        # Did the model cite one of the gold documents?
        "cites_doc":     1 if any(d.lower() in answer_lower for d in gold_doc_names) else 0,

        # Did the model admit it couldn't find the answer?
        "not_found":     1 if "not found" in answer_lower or "cannot find" in answer_lower else 0,

        # Did the model hallucinate by giving a confident answer for a missing doc?
        "hallucination_risk": 1 if (
            NUMBER_RE.search(answer) and
            not any(d.lower() in answer_lower for d in gold_doc_names)
        ) else 0,

        # Answer length (proxy for informativeness)
        "word_count": len(answer.split()),
    }


def avg_dict(lst):
    if not lst: return {}
    keys = lst[0].keys()
    return {k: round(sum(d[k] for d in lst)/len(lst), 4) for k in keys}


# ------------------------------------------------------------------
# Main evaluation
# ------------------------------------------------------------------

def evaluate(arch_ids, sample_size, gen_mode):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Check LM Studio
    print("\nChecking LM Studio connection...")
    lm_available = health_check()
    if not lm_available:
        print(
            "WARNING: LM Studio not available — retrieval metrics will still be computed,\n"
            "         but generation metrics will be skipped.\n"
            "         Start LM Studio and load qwen2.5-vl-7b-instruct to enable generation."
        )

    # Load test questions
    questions = []
    with open(TEST_SET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line: questions.append(json.loads(line))
    questions = questions[:sample_size]
    print(f"Loaded {len(questions)} questions. Loading retrieval models...")

    models = SharedModels()
    all_summary = {}

    for arch_id in arch_ids:
        arch_name, arch_module = ARCHITECTURES[arch_id]
        print(f"\n{'='*62}")
        print(f"  Architecture {arch_id}: {arch_name}")
        print(f"{'='*62}")

        retr_metrics_list = []
        gen_metrics_list  = []
        per_query_log     = []

        for qi, q in enumerate(questions):
            question      = q.get("question", "")
            gold_ids      = set(str(g) for g in q.get("gold_quote_ids", []))
            if not question or not gold_ids:
                continue

            # Retrieve
            t0 = time.time()
            try:
                result = arch_module.retrieve(question, models)
            except Exception as e:
                print(f"  ERROR Q{qi+1}: {e}")
                continue
            retr_time = time.time() - t0

            text_hits  = result.get("text",  [])
            image_hits = result.get("image", [])
            combo_ids  = list(dict.fromkeys(
                extract_quote_ids(text_hits) + extract_quote_ids(image_hits)
            ))[:3]

            rm = {
                "hit_rate":    hit_rate(combo_ids, gold_ids),
                "recall@3":    recall_at_k(combo_ids, gold_ids),
                "precision@3": precision_at_k(combo_ids, gold_ids),
                "mrr":         mrr(combo_ids, gold_ids),
                "ndcg@3":      ndcg(combo_ids, gold_ids),
                "retr_latency_s": round(retr_time, 3),
            }
            retr_metrics_list.append(rm)

            # Generate (if LM Studio available)
            answer    = ""
            gm        = {}
            gen_time  = 0.0
            gold_docs = list({str(g).split("_")[0] for g in gold_ids})

            if lm_available:
                try:
                    if gen_mode == "text_only":
                        messages = build_text_only_prompt(question, text_hits)
                    else:
                        messages = build_multimodal_prompt(question, text_hits, image_hits)

                    t1       = time.time()
                    answer   = chat(messages, temperature=0.1, max_tokens=256)
                    gen_time = time.time() - t1
                    gm       = score_answer(answer, gold_docs)
                    gm["gen_latency_s"] = round(gen_time, 3)
                    gen_metrics_list.append(gm)
                except Exception as e:
                    print(f"  Generation ERROR Q{qi+1}: {e}")

            per_query_log.append({
                "test_id":        q.get("test_id"),
                "question":       question[:100],
                "gold_ids":       list(gold_ids),
                "retrieved_ids":  combo_ids,
                "retr_metrics":   rm,
                "answer":         answer[:300] if answer else "",
                "gen_metrics":    gm,
            })

            if (qi + 1) % 10 == 0:
                print(f"  Processed {qi+1}/{len(questions)}...")

        avg_retr = avg_dict(retr_metrics_list)
        avg_gen  = avg_dict(gen_metrics_list) if gen_metrics_list else {}

        print(f"\n  Retrieval metrics:")
        for k, v in avg_retr.items():
            print(f"    {k:<20} {v:.4f}")
        if avg_gen:
            print(f"  Generation metrics:")
            for k, v in avg_gen.items():
                print(f"    {k:<20} {v:.4f}")

        arch_summary = {
            "arch_id":    arch_id,
            "arch_name":  arch_name,
            "gen_mode":   gen_mode,
            "n":          len(per_query_log),
            "retrieval":  avg_retr,
            "generation": avg_gen,
            "per_query":  per_query_log,
        }
        all_summary[arch_id] = arch_summary

        out_path = os.path.join(RESULTS_DIR, f"arch{arch_id}_gen_results.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(arch_summary, f, indent=2, ensure_ascii=False)
        print(f"  Saved: {out_path}")

    # Final comparison table
    print(f"\n{'='*70}")
    print(f"  FINAL COMPARISON  (n={sample_size}, mode={gen_mode})")
    print(f"{'='*70}")
    print(f"  {'Architecture':<28} {'NDCG@3':>7} {'MRR':>7} {'Hit@3':>7} {'HasNum':>7} {'CiteDoc':>8}")
    print(f"  {'-'*65}")
    for arch_id, res in all_summary.items():
        r = res["retrieval"]
        g = res["generation"]
        print(
            f"  {res['arch_name']:<28} "
            f"{r.get('ndcg@3',0):>7.3f} "
            f"{r.get('mrr',0):>7.3f} "
            f"{r.get('hit_rate',0):>7.3f} "
            f"{g.get('has_number',0):>7.3f} "
            f"{g.get('cites_doc',0):>8.3f}"
        )

    combined_path = os.path.join(RESULTS_DIR, "gen_comparison_summary.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "n": sample_size, "gen_mode": gen_mode,
            "architectures": {
                str(k): {"name": v["arch_name"],
                         "retrieval": v["retrieval"],
                         "generation": v["generation"]}
                for k, v in all_summary.items()
            }
        }, f, indent=2)
    print(f"\n  Full results: {combined_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int,   default=20)
    parser.add_argument("--arch",   type=int,   nargs="+", default=[1,2,3,4,5])
    parser.add_argument("--mode",   type=str,   default="multimodal",
                        choices=["text_only", "multimodal"])
    args = parser.parse_args()
    evaluate(args.arch, args.sample, args.mode)
