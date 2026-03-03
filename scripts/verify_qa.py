# scripts/verify_qa.py

import json
from pathlib import Path

QA_FILE = Path("data") / "processed" / "qa_pairs.jsonl"

count = 0
empty_q = 0
empty_candidates = 0

with open(QA_FILE, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        count += 1
        if not obj["question"]:
            empty_q += 1
        if not obj["candidate_quote_ids"]:
            empty_candidates += 1

print("Total QA records:", count)
print("Empty questions:", empty_q)
print("Empty candidate lists:", empty_candidates)

print("\nSample 3 QA records:\n")

with open(QA_FILE, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 3:
            break
        obj = json.loads(line)
        print("----")
        print("Question:", obj["question"])
        print("Candidates:", len(obj["candidate_quote_ids"]))
        print("Answer (truncated):", obj["reference_answer"][:200])     