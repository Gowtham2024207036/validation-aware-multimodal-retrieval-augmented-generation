# scripts/summary_processed.py
import json
from pathlib import Path
import csv
from itertools import islice

PROCESSED = Path("data") / "processed"
QA_FILE = PROCESSED / "qa_pairs.jsonl"
QUOTES_FILE = PROCESSED / "quotes_master.csv"

def show_qa(n=5):
    print("QA sample:")
    with open(QA_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            obj = json.loads(line)
            print(f"\n--- QA {i+1} ---")
            print("source:", obj.get("source_file"), "line:", obj.get("line_idx"))
            print("question:", obj.get("question")[:400])
            print("candidate_quote_ids:", obj.get("candidate_quote_ids")[:10])
            print("reference_answer (trunc):", (obj.get("reference_answer") or "")[:300])

def show_quotes(n=10):
    print("\nQuotes sample:")
    with open(QUOTES_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= n:
                break
            print(f"{i+1}. id:{row['quote_id']} modality:{row['modality']} src:{row['source_file']} imgpath:{row['image_path'] or '<none>'}")
            print("   text:", (row['text'] or "")[:200])

if __name__ == "__main__":
    if not QA_FILE.exists() or not QUOTES_FILE.exists():
        print("Processed files missing. Run parse script first.")
    else:
        # counts
        qa_count = sum(1 for _ in open(QA_FILE, "r", encoding="utf-8"))
        quote_count = sum(1 for _ in open(QUOTES_FILE, "r", encoding="utf-8")) - 1
        print("QA records:", qa_count)
        print("Quote rows:", quote_count)
        show_qa(5)
        show_quotes(10)