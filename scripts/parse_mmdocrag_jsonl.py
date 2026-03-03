# scripts/parse_mmdocrag_jsonl.py
"""
Parse MMDocRAG-style chat JSONL files (system/user/assistant messages)
and produce:
 - data/processed/qa_pairs.jsonl     (question, reference_answer, candidate_quote_ids, gold_ids, provenance)
 - data/processed/quotes_master.csv  (quote metadata: quote_id, modality, text, image_tag, image_path, source_file, line_idx, quote_index)
"""

import json
import re
from pathlib import Path
import csv

RAW_DIR = Path("data") / "raw"
OUT_DIR = Path("data") / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

QA_OUT = OUT_DIR / "qa_pairs.jsonl"
QUOTES_OUT = OUT_DIR / "quotes_master.csv"

# regexes
# capture numbered text quotes like:
# [1] some text... [2] next text...
TEXT_QUOTE_PATTERN = re.compile(r'\[\s*(\d+)\s*\]\s*(.*?)\s*(?=(?:\[\s*\d+\s*\]\s*)|(?:Image Quotes are:)|$)', re.DOTALL | re.IGNORECASE)
# image descriptions: "image1 is described as: <text>" repeated
IMAGE_QUOTE_PATTERN = re.compile(r'(image\d+)\s+(?:is described as:|:)\s*(.*?)(?=(?:image\d+\s+(?:is described as:|:))|(?:The user question is:)|$)', re.DOTALL | re.IGNORECASE)
# question pattern
QUESTION_PATTERNS = [
    re.compile(r'The user question is[:\s]*([^\n].*?)\s*$', re.IGNORECASE | re.DOTALL),
    re.compile(r'Question\s*[:\s]*(.+)', re.IGNORECASE),
    re.compile(r'Q\s*[:\s]*(.+)', re.IGNORECASE)
]
# fallback: last line ending with '?'
LAST_QUESTION_FALLBACK = re.compile(r'([^\n?]+\?)\s*$', re.DOTALL)

def extract_text_quotes(user_content):
    """Return list of (index:int, text:str)"""
    matches = TEXT_QUOTE_PATTERN.findall(user_content)
    # TEXT_QUOTE_PATTERN returns list of (index, text)
    out = []
    for idx_str, txt in matches:
        idx = int(idx_str)
        out.append((idx, txt.strip()))
    return out

def extract_image_quotes(user_content):
    """Return list of (image_tag, text)"""
    matches = IMAGE_QUOTE_PATTERN.findall(user_content)
    out = []
    for tag, desc in matches:
        out.append((tag.strip(), desc.strip()))
    return out

def extract_question(user_content, assistant_content):
    # 1) try explicit patterns in user content
    for pat in QUESTION_PATTERNS:
        m = pat.search(user_content)
        if m:
            return m.group(1).strip()
    # 2) try assistant content (maybe the assistant echoed question)
    for pat in QUESTION_PATTERNS:
        m = pat.search(assistant_content or "")
        if m:
            return m.group(1).strip()
    # 3) fallback: last sentence in user_content ending with '?'
    m = LAST_QUESTION_FALLBACK.search(user_content)
    if m:
        return m.group(1).strip()
    # 4) not found
    return None

def normalize_quote_id(source_file_stem, line_idx, modality, qindex):
    # modality in {'text','image','table','unknown'}
    return f"{source_file_stem}_L{line_idx}_{modality[:1].upper()}{qindex}"

def locate_image_path(image_tag, raw_images_dir):
    """
    Attempt to match image_tag (like image1) to a file in raw_images_dir.
    We'll search for files that contain the tag as substring (case-insensitive).
    If not found, return empty string.
    """
    if not raw_images_dir.exists():
        return ""
    for p in raw_images_dir.rglob("*"):
        if not p.is_file():
            continue
        name = p.name.lower()
        if image_tag.lower() in name:
            return str(p)
    return ""

def process_file(path: Path, qa_writer, quote_rows, images_root):
    """
    path: path to .jsonl file
    qa_writer: file object to write JSON lines to
    quote_rows: list to append quote dicts for csv
    images_root: Path to data/raw/images (or another folder)
    """
    stem = path.stem
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"[WARN] Failed to parse JSON in {path} line {line_idx}: {e}")
                continue
            messages = obj.get("messages", [])
            # prefer user message at index 1 if available
            user_content = ""
            assistant_content = ""
            if len(messages) >= 2:
                user_content = messages[1].get("content", "") or ""
            if len(messages) >= 3:
                assistant_content = messages[2].get("content", "") or ""
            # extract quotes and question
            text_quotes = extract_text_quotes(user_content)  # list of (idx, text)
            image_quotes = extract_image_quotes(user_content)  # list of (imageTag, desc)
            question = extract_question(user_content, assistant_content)
            # if question missing, store empty string but continue
            # write QA record
            candidate_ids = []
            gold_ids = []  # we may look for explicit "Gold Quotes:" later (not mandatory)
            # make unique deterministic quote ids and add to quote_rows
            # text quotes
            for qidx, (num, txt) in enumerate(text_quotes, start=1):
                quote_id = normalize_quote_id(stem, line_idx, "text", qidx)
                candidate_ids.append(quote_id)
                quote_rows.append({
                    "quote_id": quote_id,
                    "modality": "text",
                    "text": txt,
                    "image_tag": "",
                    "image_path": "",
                    "source_file": path.name,
                    "line_idx": line_idx,
                    "quote_index": qidx
                })
            # image quotes
            for qidx, (img_tag, desc) in enumerate(image_quotes, start=1):
                quote_id = normalize_quote_id(stem, line_idx, "image", qidx)
                candidate_ids.append(quote_id)
                # try to locate local image file under images_root
                image_path = locate_image_path(img_tag, images_root)
                quote_rows.append({
                    "quote_id": quote_id,
                    "modality": "image",
                    "text": desc,
                    "image_tag": img_tag,
                    "image_path": image_path,
                    "source_file": path.name,
                    "line_idx": line_idx,
                    "quote_index": qidx
                })
            # assemble QA record
            qa_rec = {
                "source_file": path.name,
                "line_idx": line_idx,
                "question": question or "",
                "reference_answer": assistant_content.strip(),
                "candidate_quote_ids": candidate_ids,
                "gold_quote_ids": gold_ids  # left empty if not present; we may fill later from other sources
            }
            qa_writer.write(json.dumps(qa_rec, ensure_ascii=False) + "\n")

def main():
    raw_files = sorted([p for p in RAW_DIR.glob("*.jsonl")])
    if not raw_files:
        print("No .jsonl files found in data/raw/. Place train.jsonl and dev/eval splits there.")
        return
    images_root = RAW_DIR / "images"
    all_quote_rows = []
    with open(QA_OUT, "w", encoding="utf-8") as qa_f:
        for p in raw_files:
            print(f"Processing {p} ...")
            process_file(p, qa_f, all_quote_rows, images_root)

    # write quotes CSV
    with open(QUOTES_OUT, "w", encoding="utf-8", newline="") as csvf:
        fieldnames = ["quote_id", "modality", "text", "image_tag", "image_path", "source_file", "line_idx", "quote_index"]
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_quote_rows:
            writer.writerow(row)

    print("Wrote QA records to:", QA_OUT)
    print("Wrote quotes master to:", QUOTES_OUT)
    print(f"Total QA records: (see lines in {QA_OUT})")
    print(f"Total quotes rows: {len(all_quote_rows)}")

if __name__ == "__main__":
    main()