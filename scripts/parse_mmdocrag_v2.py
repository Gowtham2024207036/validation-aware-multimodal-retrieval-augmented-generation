# scripts/parse_mmdocrag_v3.py
import json
import re
from pathlib import Path
import csv

try:
    import pandas as pd
except Exception as e:
    raise ImportError("pandas is required for this script. Install with: pip install pandas") from e

RAW_DIR = Path("data") / "raw"
OUT_DIR = Path("data") / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

QA_OUT = OUT_DIR / "qa_pairs.jsonl"
QUOTES_JSONL_OUT = OUT_DIR / "quotes_master.jsonl"
QUOTES_CSV_OUT = OUT_DIR / "quotes_master.csv"

# ---------- EXTRACTION FUNCTIONS ----------

def extract_text_quotes(content):
    """
    Extract numbered text quotes under 'Text Quotes are:' section
    """
    if "Text Quotes are:" not in content:
        return []

    text_section = content.split("Text Quotes are:")[1]

    # Stop at Image Quotes section if exists
    if "Image Quotes are:" in text_section:
        text_section = text_section.split("Image Quotes are:")[0]

    # Use a more inclusive pattern that tolerates newlines and unusual punctuation
    pattern = re.compile(r"\[\s*(\d+)\s*\]\s*(.*?)(?=(?:\n\[\s*\d+\s*\])|$)", re.DOTALL)
    matches = pattern.findall(text_section)

    quotes = []
    for idx, txt in matches:
        quotes.append((int(idx), txt.strip()))
    return quotes


def extract_image_quotes(content):
    """
    Extract image descriptions like:
    image1 is described as: ...
    """
    if "Image Quotes are:" not in content:
        return []

    image_section = content.split("Image Quotes are:")[1]

    pattern = re.compile(
        r"(image\d+)\s+(?:is described as:|:)\s*(.*?)(?=(?:image\d+\s+(?:is described as:|:))|(?:The user question is:)|$)",
        re.DOTALL | re.IGNORECASE,
    )
    matches = pattern.findall(image_section)

    images = []
    for tag, desc in matches:
        images.append((tag.strip(), desc.strip()))
    return images


def extract_question(content):
    """
    Extract question from 'The user question is:' pattern
    """
    # Attempt multiple heuristics to find the question
    patterns = [
        re.compile(r"The user question is:\s*(.*)", re.IGNORECASE | re.DOTALL),
        re.compile(r"User question\s*[:\-]\s*(.*)", re.IGNORECASE | re.DOTALL),
        re.compile(r"Question\s*[:\-]\s*(.*)", re.IGNORECASE | re.DOTALL),
    ]
    for pat in patterns:
        m = pat.search(content)
        if m:
            # take first line or sentence if it's long
            q = m.group(1).strip()
            # sometimes assistant answer appended; split at "The user question is:" unlikely, but trim at newline if multiple paragraphs
            q = q.split("\n")[0].strip()
            return q
    # fallback: last sentence that ends with '?'
    m = re.search(r'([^\n?]+\?)\s*$', content)
    if m:
        return m.group(1).strip()
    return ""


def normalize_quote_id(source, line_idx, modality, index):
    return f"{source}_L{line_idx}_{modality}_{index}"


def process_file(path: Path, qa_writer, quote_rows):
    stem = path.stem

    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"[WARN] Failed to parse JSON in {path} line {line_idx}: {e}")
                continue

            messages = obj.get("messages", [])

            if len(messages) < 2:
                # skip malformed
                continue

            user_content = messages[1].get("content", "") or ""
            assistant_content = messages[2].get("content", "") if len(messages) > 2 else ""

            text_quotes = extract_text_quotes(user_content)
            image_quotes = extract_image_quotes(user_content)
            question = extract_question(user_content)

            candidate_ids = []

            # TEXT QUOTES
            for i, (num, txt) in enumerate(text_quotes, start=1):
                qid = normalize_quote_id(stem, line_idx, "text", i)
                candidate_ids.append(qid)
                quote_rows.append({
                    "quote_id": qid,
                    "modality": "text",
                    "text": txt,
                    "image_tag": "",
                    "image_path": "",
                    "source_file": path.name,
                    "line_idx": line_idx,
                    "quote_index": i
                })

            # IMAGE QUOTES
            for i, (tag, desc) in enumerate(image_quotes, start=1):
                qid = normalize_quote_id(stem, line_idx, "image", i)
                candidate_ids.append(qid)
                quote_rows.append({
                    "quote_id": qid,
                    "modality": "image",
                    "text": desc,
                    "image_tag": tag,
                    "image_path": "",
                    "source_file": path.name,
                    "line_idx": line_idx,
                    "quote_index": i
                })

            qa_record = {
                "source_file": path.name,
                "line_idx": line_idx,
                "question": question,
                "reference_answer": assistant_content.strip(),
                "candidate_quote_ids": candidate_ids,
                "gold_quote_ids": []
            }

            qa_writer.write(json.dumps(qa_record, ensure_ascii=False) + "\n")


def main():
    raw_files = sorted(RAW_DIR.glob("*.jsonl"))

    if not raw_files:
        print("No jsonl files found in data/raw/")
        return

    quote_rows = []

    with open(QA_OUT, "w", encoding="utf-8") as qa_writer:
        for file in raw_files:
            print(f"Processing {file} ...")
            process_file(file, qa_writer, quote_rows)

    # write jsonl of quotes (safe)
    with open(QUOTES_JSONL_OUT, "w", encoding="utf-8") as qj:
        for row in quote_rows:
            qj.write(json.dumps(row, ensure_ascii=False) + "\n")

    # write CSV robustly using pandas
    try:
        df = pd.DataFrame(quote_rows)
        # ensure consistent columns even if empty
        expected_cols = ["quote_id", "modality", "text", "image_tag", "image_path", "source_file", "line_idx", "quote_index"]
        for c in expected_cols:
            if c not in df.columns:
                df[c] = ""
        df.to_csv(QUOTES_CSV_OUT, index=False, quoting=csv.QUOTE_ALL)
        print(f"Wrote CSV: {QUOTES_CSV_OUT}")
    except Exception as e:
        print(f"[WARN] pandas CSV write failed: {e}")
        print(f"Falling back to writing quotes JSONL only at {QUOTES_JSONL_OUT}")

    print("Parsing complete.")
    print("QA file:", QA_OUT)
    print("Quotes jsonl:", QUOTES_JSONL_OUT)
    print("Quotes csv (attempted):", QUOTES_CSV_OUT)
    print("Total quotes:", len(quote_rows))


if __name__ == "__main__":
    main()