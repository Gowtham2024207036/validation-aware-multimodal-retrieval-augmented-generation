# scripts/inspect_single_full.py

import json
from pathlib import Path

DATA_PATH = Path("data")
FILE = DATA_PATH / "train.jsonl"

with open(FILE, "r", encoding="utf-8") as f:
    first_line = f.readline()
    obj = json.loads(first_line)

messages = obj["messages"]

print("\n===== USER MESSAGE FULL CONTENT =====\n")
print(messages[1]["content"])