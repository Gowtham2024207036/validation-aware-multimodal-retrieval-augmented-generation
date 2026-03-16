"""
prepare_validation_data.py  —  Fixed for both file formats
-----------------------------------------------------------
MMDocRAG has TWO different formats:

FORMAT A — train.jsonl  (messages array):
  {"messages": [{"role":"system",...}, {"role":"user",...}, {"role":"assistant",...}]}
  Labels derived from citations in the assistant response.

FORMAT B — dev_20.jsonl / evaluation_20.jsonl  (flat format):
  {"q_id":0, "doc_name":"...", "question":"...",
   "text_quotes":[{"quote_id":"text1","text":"..."},...],
   "img_quotes": [{"quote_id":"image1","img_description":"..."},...],
   "gold_quotes":["text5","image7"],
   "answer_short":"0.16", "answer_interleaved":"..."}
  Labels derived directly from gold_quotes list.

Label rules (both formats):
  Label 2 = quote appears in gold answer  → cited / highly relevant
  Label 1 = quote present but NOT cited   → candidate, not relevant

Usage:
    python scripts/prepare_validation_data.py
    python scripts/prepare_validation_data.py --debug
    python scripts/prepare_validation_data.py --max_records 500  # quick test
"""

import os, sys, re, json, argparse, logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEXT_CITE_RE  = re.compile(r"\[(\d+)\]")
IMAGE_CITE_RE = re.compile(r"!\[.*?\]\(image(\d+)\)", re.IGNORECASE)
IMAGE_BARE_RE = re.compile(r"\bimage\s*(\d+)\b",      re.IGNORECASE)


# ================================================================== #
#  FORMAT B  —  flat  (dev / eval)
# ================================================================== #

def process_flat_record(record: dict) -> list[dict]:
    """
    Handles dev_20.jsonl and evaluation_20.jsonl.
    gold_quotes is a list like ["text5", "image7"] — direct label lookup.
    """
    question = record.get("question", "").strip()
    if not question:
        return []

    gold_set = set(record.get("gold_quotes", []))
    doc_name = record.get("doc_name", "")
    pairs    = []

    for tq in record.get("text_quotes", []):
        qid  = tq.get("quote_id", "")
        text = tq.get("text", "").strip()
        if not text:
            continue
        pairs.append({
            "question": question,
            "chunk":    text,
            "modality": "text",
            "quote_id": qid,
            "label":    2 if qid in gold_set else 1,
            "doc_name": doc_name,
        })

    for iq in record.get("img_quotes", []):
        qid  = iq.get("quote_id", "")
        desc = iq.get("img_description", "").strip()
        if not desc:
            continue
        pairs.append({
            "question": question,
            "chunk":    desc,
            "modality": "image",
            "quote_id": qid,
            "label":    2 if qid in gold_set else 1,
            "doc_name": doc_name,
        })

    return pairs


# ================================================================== #
#  FORMAT A  —  messages  (train)
# ================================================================== #

def parse_user_msg(content: str) -> dict:
    text_quotes, image_quotes = {}, {}
    cur_idx, cur_lines, in_tq = None, [], False

    for line in content.split("\n"):
        m = re.match(r"^\[(\d+)\]\s*(.*)", line)
        if m:
            if cur_idx is not None:
                text_quotes[cur_idx] = " ".join(cur_lines).strip()
            cur_idx, cur_lines, in_tq = int(m.group(1)), [m.group(2)] if m.group(2).strip() else [], True
            continue
        if re.match(r"^image\d+\s+is\s+described|^Image Quotes", line, re.IGNORECASE) or \
           re.match(r"The user question is:|User question is:", line, re.IGNORECASE):
            if cur_idx is not None:
                text_quotes[cur_idx] = " ".join(cur_lines).strip()
                cur_idx = None
            in_tq = False
            break
        if in_tq and cur_idx is not None:
            cur_lines.append(line)
    if cur_idx is not None:
        text_quotes[cur_idx] = " ".join(cur_lines).strip()

    for m in re.finditer(
        r"image(\d+)\s+is\s+described\s+as:\s*(.*?)(?=\nimage\d+\s+is\s+described|\nThe user question|\Z)",
        content, re.DOTALL | re.IGNORECASE
    ):
        image_quotes[int(m.group(1))] = m.group(2).strip()

    question = ""
    qm = re.search(r"(?:The user question is:|User question is:)\s*(.*?)(?:\n|$)", content, re.IGNORECASE)
    if qm:
        question = qm.group(1).strip()
    if not question:
        ls = [l.strip() for l in content.splitlines() if l.strip()]
        question = ls[-1] if ls else ""

    return {"question": question, "text_quotes": text_quotes, "image_quotes": image_quotes}


def parse_citations(assistant: str) -> tuple[set, set]:
    cited_t = set(int(m) for m in TEXT_CITE_RE.findall(assistant))
    cited_i = set(int(m) for m in IMAGE_CITE_RE.findall(assistant))
    if not cited_i:
        cited_i = set(int(m) for m in IMAGE_BARE_RE.findall(assistant))
    return cited_t, cited_i


def process_messages_record(record: dict) -> list[dict]:
    """Handles train.jsonl messages format."""
    user_msg = asst_msg = ""
    for msg in record.get("messages", []):
        r, c = msg.get("role",""), msg.get("content","")
        if r == "user"      and not user_msg: user_msg = c
        elif r == "assistant" and not asst_msg: asst_msg = c

    if not user_msg or not asst_msg:
        return []

    parsed   = parse_user_msg(user_msg)
    question = parsed["question"]
    if not question:
        return []

    cited_t, cited_i = parse_citations(asst_msg)
    doc_name = record.get("doc_name", "")
    pairs    = []

    for idx, chunk in parsed["text_quotes"].items():
        if not chunk.strip(): continue
        pairs.append({
            "question": question, "chunk": chunk.strip(),
            "modality": "text",   "quote_id": f"text{idx}",
            "label":    2 if idx in cited_t else 1,
            "doc_name": doc_name,
        })
    for idx, desc in parsed["image_quotes"].items():
        if not desc.strip(): continue
        pairs.append({
            "question": question, "chunk": desc.strip(),
            "modality": "image",  "quote_id": f"image{idx}",
            "label":    2 if idx in cited_i else 1,
            "doc_name": doc_name,
        })
    return pairs


# ================================================================== #
#  Auto-detect format
# ================================================================== #

def detect_format(record: dict) -> str:
    if "messages" in record:                          return "messages"
    if "text_quotes" in record or "img_quotes" in record: return "flat"
    return "unknown"


def process_record(record: dict) -> list[dict]:
    fmt = detect_format(record)
    if fmt == "flat":     return process_flat_record(record)
    if fmt == "messages": return process_messages_record(record)
    return []


# ================================================================== #
#  File processor
# ================================================================== #

def process_file(fpath: Path, out_path: Path, max_records: int = None) -> dict:
    if not fpath.exists():
        log.warning("File not found: %s", fpath)
        return {"n_pairs": 0}

    pairs, ok, skip = [], 0, 0
    fmt_counts = {}

    with open(fpath, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if max_records and ln > max_records: break
            line = line.strip()
            if not line: continue
            try:    record = json.loads(line)
            except: skip += 1; continue

            fmt = detect_format(record)
            fmt_counts[fmt] = fmt_counts.get(fmt, 0) + 1

            recs = process_record(record)
            if recs: pairs.extend(recs); ok += 1
            else:    skip += 1

            if ln % 1000 == 0:
                log.info("  %s: %d records → %d pairs", fpath.name, ln, len(pairs))

    lc = {1: 0, 2: 0}
    mc = {"text": 0, "image": 0}
    for p in pairs:
        lc[p["label"]]    = lc.get(p["label"], 0) + 1
        mc[p["modality"]] = mc.get(p["modality"], 0) + 1

    with open(out_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    log.info("  %s → %d pairs | labels: L1=%d L2=%d | text=%d image=%d | fmt=%s",
             fpath.name, len(pairs), lc.get(1,0), lc.get(2,0),
             mc.get("text",0), mc.get("image",0), fmt_counts)

    if lc.get(2, 0) == 0:
        log.warning("  WARNING: 0 label-2 (gold) samples! Check file format.")

    return {"n_pairs": len(pairs), "labels": lc}


# ================================================================== #
#  Debug helper
# ================================================================== #

def debug_file(fpath: Path):
    log.info("=== DEBUG: %s ===", fpath)
    with open(fpath, "r", encoding="utf-8") as f:
        line = f.readline().strip()
    record = json.loads(line)
    fmt    = detect_format(record)
    log.info("Format detected: %s", fmt)
    log.info("Top-level keys:  %s", list(record.keys()))

    if fmt == "flat":
        log.info("question:    %s", record.get("question","")[:80])
        log.info("text_quotes: %d items", len(record.get("text_quotes",[])))
        log.info("img_quotes:  %d items", len(record.get("img_quotes",[])))
        log.info("gold_quotes: %s", record.get("gold_quotes",[]))
    else:
        msgs = record.get("messages",[])
        log.info("messages:    %d items", len(msgs))
        for m in msgs:
            log.info("  role=%s content_len=%d", m.get("role"), len(m.get("content","")))

    pairs = process_record(record)
    lc    = {1:0, 2:0}
    for p in pairs: lc[p["label"]] = lc.get(p["label"],0) + 1
    log.info("Test parse: %d pairs, labels: %s", len(pairs), lc)
    if pairs:
        log.info("Sample [label=%d]: %s...", pairs[0]["label"], pairs[0]["chunk"][:80])


# ================================================================== #
#  Main
# ================================================================== #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug",       action="store_true",
                        help="Inspect first record of each file")
    parser.add_argument("--max_records", type=int, default=None,
                        help="Limit records per file (quick test)")
    args = parser.parse_args()

    splits = [
        ("train", ["train.jsonl"]),
        ("dev",   ["dev_20.jsonl", "dev_15.jsonl"]),
        ("eval",  ["evaluation_20.jsonl", "evaluation_15.jsonl",
                   "eval_20.jsonl", "eval_15.jsonl"]),
    ]

    for split_name, candidates in splits:
        fpath = next((RAW_DIR / f for f in candidates if (RAW_DIR / f).exists()), None)
        if fpath is None:
            log.warning("Skipping '%s' — none of %s found in %s",
                        split_name, candidates, RAW_DIR)
            continue

        log.info("\n--- Split: %-6s  (%s) ---", split_name, fpath.name)
        if args.debug:
            debug_file(fpath)

        out_path = OUT_DIR / f"validation_pairs_{split_name}.jsonl"
        process_file(fpath, out_path, max_records=args.max_records)

    log.info("\nAll done. Files in: %s", OUT_DIR.resolve())


if __name__ == "__main__":
    main()