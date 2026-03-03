# # scripts/inspect_dataset.py

# import json
# from pathlib import Path

# DATA_PATH = Path("data")

# def inspect_file(filename):
#     file_path = DATA_PATH / filename
    
#     print(f"\nInspecting: {file_path}")
#     print("-" * 50)
    
#     with open(file_path, "r", encoding="utf-8") as f:
#         first_line = f.readline()
#         sample = json.loads(first_line)
    
#     print("Top-level keys:")
#     for key in sample.keys():
#         print(f"  - {key}")
    
#     print("\nSample Entry (truncated to 1000 chars):")
#     print(str(sample)[:1000])

# if __name__ == "__main__":
#     inspect_file("train.jsonl")


# scripts/inspect_more.py
import json
import re
from pathlib import Path
from textwrap import shorten

DATA_PATH = Path("data")
INPUT_FILE = DATA_PATH / "train.jsonl"
N = 5  # how many records to inspect

# heuristic regexes to identify common patterns
PATTERNS = {
    "question_like": re.compile(r'(?i)\bquestion\b[:\s]*["\']?(.*?)(?:["\']?$|\n|$)', re.DOTALL),
    "q_colon": re.compile(r'Q\s*[:\-]\s*(.+)', re.IGNORECASE),
    "gold_quotes": re.compile(r'(?i)gold quotes[:\s]*([\[\{].*?[\]\}])', re.DOTALL),
    "text_quotes": re.compile(r'(?i)text quotes[:\s]*([\[\{].*?[\]\}])', re.DOTALL),
    "image_quotes": re.compile(r'(?i)image quotes[:\s]*([\[\{].*?[\]\}])', re.DOTALL),
    "candidate_quotes": re.compile(r'(?i)candidate quotes[:\s]*([\[\{].*?[\]\}])', re.DOTALL),
    "quotes_map": re.compile(r'(?m)\"(text\d+)\"\s*:\s*\"(.+?)\"'),
}

def safe_truncate(s, width=1000):
    return shorten(s.replace("\r", " "), width=width, placeholder=" ... [TRUNC]")

def try_json_parse(s):
    try:
        return json.loads(s)
    except Exception:
        return None

def inspect():
    print(f"Inspecting first {N} records of {INPUT_FILE}")
    print("=" * 80)
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= N:
                break
            print(f"\n--- RECORD {i+1} ---")
            try:
                obj = json.loads(line)
            except Exception as e:
                print("Failed to parse JSON line:", e)
                print(safe_truncate(line, 800))
                continue

            # top-level keys
            print("Top-level keys:", list(obj.keys()))
            # if 'messages' present, print length and roles summary
            if "messages" in obj and isinstance(obj["messages"], list):
                print("messages length:", len(obj["messages"]))
                for j, m in enumerate(obj["messages"][:6]):
                    role = m.get("role", "<no role>")
                    content = m.get("content", "")
                    print(f"  msg[{j}] role={role} content(trunc): {safe_truncate(content, 250)}")
            else:
                # dump a truncated representation
                print("Object preview:", safe_truncate(json.dumps(obj, ensure_ascii=False), 800))

            # apply heuristics on the combined string of message contents
            all_text = ""
            if "messages" in obj and isinstance(obj["messages"], list):
                all_text = "\n\n".join([m.get("content","") for m in obj["messages"]])
            else:
                all_text = json.dumps(obj, ensure_ascii=False)

            # run pattern detection
            for name, pat in PATTERNS.items():
                m = pat.search(all_text)
                if m:
                    found = safe_truncate(m.group(1) if m.groups() else m.group(0), 300)
                    print(f"  [DETECTED: {name}] -> {found}")

            # try to find inline quote map (text1: "....")
            maps = PATTERNS["quotes_map"].findall(all_text)
            if maps:
                print(f"  Found {len(maps)} inline quote mappings (showing up to 5):")
                for k, v in maps[:5]:
                    print(f"    {k} => {safe_truncate(v, 150)}")
            else:
                print("  No simple inline 'textX': '...' mappings detected in this record.")

    print("\nDone inspection.")

if __name__ == "__main__":
    inspect()