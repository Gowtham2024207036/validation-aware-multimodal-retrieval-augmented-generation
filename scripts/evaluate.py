"""
evaluate.py  —  Evaluation Harness for all 5 RAG Architectures
---------------------------------------------------------------
Metrics computed per architecture:
  - Hit Rate    : fraction of queries where ≥1 gold quote is in top-3
  - Recall@3    : fraction of gold quotes retrieved in top-3
  - Precision@3 : fraction of top-3 results that are gold quotes
  - MRR         : Mean Reciprocal Rank (1/rank of first correct hit)
  - NDCG@3      : Normalised Discounted Cumulative Gain at rank 3

Each metric is computed separately for text, image, and combined results.
All 5 architectures run on the same test_set.jsonl questions.

Usage:
    python evaluate.py --sample 20        # quick test on 20 questions
    python evaluate.py --sample 150       # full evaluation
    python evaluate.py --arch 1 2         # run only arch 1 and 2
"""

import os
import sys
import json
import math
import argparse
import time
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_retriever import SharedModels
import arch1_naive
import arch2_hybrid_rrf
import arch3_metadata_filter
import arch4_late_fusion
import arch5_query_expansion

TEST_SET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "raw", "test_set.jsonl"
)
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "results"
)

ARCHITECTURES = {
    1: ("Naive Multimodal RAG",       arch1_naive),
    2: ("Hybrid RAG + RRF",           arch2_hybrid_rrf),
    3: ("Metadata-Filtered RAG",      arch3_metadata_filter),
    4: ("Late Fusion RAG",            arch4_late_fusion),
    5: ("Query Expansion RAG",        arch5_query_expansion),
}


# ------------------------------------------------------------------
# Metric helpers
# ------------------------------------------------------------------

def hit_rate(retrieved_ids: list[str], gold_ids: set[str]) -> float:
    return 1.0 if any(r in gold_ids for r in retrieved_ids) else 0.0


def recall_at_k(retrieved_ids: list[str], gold_ids: set[str]) -> float:
    if not gold_ids:
        return 0.0
    hits = sum(1 for r in retrieved_ids if r in gold_ids)
    return hits / len(gold_ids)


def precision_at_k(retrieved_ids: list[str], gold_ids: set[str]) -> float:
    if not retrieved_ids:
        return 0.0
    hits = sum(1 for r in retrieved_ids if r in gold_ids)
    return hits / len(retrieved_ids)


def mrr(retrieved_ids: list[str], gold_ids: set[str]) -> float:
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in gold_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int = 3) -> float:
    dcg  = sum(
        1.0 / math.log2(rank + 1)
        for rank, rid in enumerate(retrieved_ids[:k], start=1)
        if rid in gold_ids
    )
    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, min(len(gold_ids), k) + 1)
    )
    return dcg / idcg if idcg > 0 else 0.0


def extract_quote_ids(hits: list[dict]) -> list[str]:
    """Extract quote_id from result payloads."""
    ids = []
    for h in hits:
        qid = h.get("quote_id")
        if qid:
            ids.append(str(qid))
    return ids


def compute_metrics(retrieved: list[str], gold: set[str]) -> dict:
    return {
        "hit_rate":     hit_rate(retrieved, gold),
        "recall@3":     recall_at_k(retrieved, gold),
        "precision@3":  precision_at_k(retrieved, gold),
        "mrr":          mrr(retrieved, gold),
        "ndcg@3":       ndcg_at_k(retrieved, gold),
    }


def average_metrics(metric_list: list[dict]) -> dict:
    if not metric_list:
        return {}
    keys = metric_list[0].keys()
    return {k: round(sum(m[k] for m in metric_list) / len(metric_list), 4) for k in keys}


# ------------------------------------------------------------------
# Main evaluation loop
# ------------------------------------------------------------------

def evaluate(arch_ids: list[int], sample_size: int):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load test questions
    questions = []
    with open(TEST_SET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))

    questions = questions[:sample_size]
    print(f"\nLoaded {len(questions)} test questions.")

    # Load models once (shared across all architectures)
    print("\nLoading shared models (runs once)...")
    models = SharedModels()

    all_results = {}

    for arch_id in arch_ids:
        arch_name, arch_module = ARCHITECTURES[arch_id]
        print(f"\n{'='*62}")
        print(f"  Architecture {arch_id}: {arch_name}")
        print(f"{'='*62}")

        text_metrics_list  = []
        image_metrics_list = []
        combo_metrics_list = []
        latency_list       = []
        per_query_log      = []

        for qi, q in enumerate(questions):
            question    = q.get("question", "")
            gold_ids    = set(str(g) for g in q.get("gold_quote_ids", []))

            if not question or not gold_ids:
                continue

            t0 = time.time()
            try:
                result = arch_module.retrieve(question, models)
            except Exception as e:
                print(f"  ERROR on Q{qi+1}: {e}")
                continue
            latency = time.time() - t0

            text_ids  = extract_quote_ids(result.get("text",  []))
            image_ids = extract_quote_ids(result.get("image", []))
            combo_ids = list(dict.fromkeys(text_ids + image_ids))[:3]

            tm = compute_metrics(text_ids,  gold_ids)
            im = compute_metrics(image_ids, gold_ids)
            cm = compute_metrics(combo_ids, gold_ids)

            text_metrics_list.append(tm)
            image_metrics_list.append(im)
            combo_metrics_list.append(cm)
            latency_list.append(latency)

            per_query_log.append({
                "test_id":      q.get("test_id"),
                "question":     question[:80],
                "gold_ids":     list(gold_ids),
                "text_ids":     text_ids,
                "image_ids":    image_ids,
                "text_metrics": tm,
                "image_metrics":im,
                "combo_metrics":cm,
                "latency_s":    round(latency, 3),
            })

            if (qi + 1) % 10 == 0:
                print(f"  Processed {qi+1}/{len(questions)} questions...")

        avg_text  = average_metrics(text_metrics_list)
        avg_image = average_metrics(image_metrics_list)
        avg_combo = average_metrics(combo_metrics_list)
        avg_lat   = round(sum(latency_list) / len(latency_list), 3) if latency_list else 0

        print(f"\n  Results for Architecture {arch_id}: {arch_name}")
        print(f"  {'Metric':<15} {'Text':>8} {'Image':>8} {'Combined':>10}")
        print(f"  {'-'*43}")
        for metric in ["hit_rate", "recall@3", "precision@3", "mrr", "ndcg@3"]:
            print(f"  {metric:<15} {avg_text.get(metric,0):>8.4f} "
                  f"{avg_image.get(metric,0):>8.4f} "
                  f"{avg_combo.get(metric,0):>10.4f}")
        print(f"  {'avg latency':<15} {avg_lat:>8.3f}s")

        arch_result = {
            "arch_id":    arch_id,
            "arch_name":  arch_name,
            "n_questions":len(per_query_log),
            "avg_text":   avg_text,
            "avg_image":  avg_image,
            "avg_combo":  avg_combo,
            "avg_latency_s": avg_lat,
            "per_query":  per_query_log,
        }
        all_results[arch_id] = arch_result

        # Save individual architecture results
        out_path = os.path.join(RESULTS_DIR, f"arch{arch_id}_results.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(arch_result, f, indent=2)
        print(f"  Saved to {out_path}")

    # Save combined summary
    summary = {
        "timestamp":    datetime.now().isoformat(),
        "n_questions":  sample_size,
        "architectures": {
            str(k): {
                "name":      v["arch_name"],
                "combined":  v["avg_combo"],
                "text":      v["avg_text"],
                "image":     v["avg_image"],
                "latency_s": v["avg_latency_s"],
            }
            for k, v in all_results.items()
        }
    }
    summary_path = os.path.join(RESULTS_DIR, "comparison_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Print final comparison table
    print(f"\n{'='*62}")
    print(f"  FINAL COMPARISON SUMMARY  (n={sample_size} questions)")
    print(f"{'='*62}")
    print(f"  {'Architecture':<30} {'Hit@3':>6} {'Recall':>7} {'Prec':>6} {'MRR':>7} {'NDCG':>7} {'ms':>6}")
    print(f"  {'-'*60}")
    for arch_id, res in all_results.items():
        c = res["avg_combo"]
        print(
            f"  {res['arch_name']:<30} "
            f"{c.get('hit_rate',0):>6.3f} "
            f"{c.get('recall@3',0):>7.3f} "
            f"{c.get('precision@3',0):>6.3f} "
            f"{c.get('mrr',0):>7.3f} "
            f"{c.get('ndcg@3',0):>7.3f} "
            f"{res['avg_latency_s']*1000:>6.0f}"
        )
    print(f"\n  Full results saved to: {RESULTS_DIR}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG architectures")
    parser.add_argument("--sample", type=int, default=50,
                        help="Number of test questions to evaluate (default: 50)")
    parser.add_argument("--arch",   type=int, nargs="+", default=[1,2,3,4,5],
                        help="Architecture IDs to run (default: all 5)")
    args = parser.parse_args()

    evaluate(arch_ids=args.arch, sample_size=args.sample)
