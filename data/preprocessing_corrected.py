# # import json
# # import os
# # import re

# # input_files = ["data/raw/train.jsonl", "data/raw/dev_15.jsonl", "data/raw/dev_20.jsonl", "data/raw/evaluation_15.jsonl", "data/raw/evaluation_20.jsonl"]
# # quotes_out = "data/processed/quotes_master.jsonl"
# # qa_out = "data/processed/qa_pairs.jsonl"

# # seen_quotes = set()
# # master_quotes = []
# # qa_pairs = []

# # print("🚀 Starting Surgical MMDocRAG Refactoring...")

# # for file_name in input_files:
# #     if not os.path.exists(file_name):
# #         print(f"Skipping {file_name} (not found)")
# #         continue
        
# #     with open(file_name, 'r', encoding='utf-8') as f:
# #         for line in f:
# #             data = json.loads(line)
# #             doc_name = data.get("doc_name", "unknown")
            
# #             # --- 1. Surgical Question Extraction ---
# #             user_content = ""
# #             ref_answer = ""
# #             for msg in data.get("messages", []):
# #                 if msg["role"] == "user":
# #                     user_content = msg["content"]
# #                 if msg["role"] == "assistant":
# #                     ref_answer = msg["content"]

# #             # Extract only the actual question at the end
# #             if "The user question is:" in user_content:
# #                 clean_question = user_content.split("The user question is:")[-1].strip()
# #             else:
# #                 # If the string isn't there, just take the last line
# #                 clean_question = user_content.split('\n')[-1].strip()

# #             # --- 2. Process Text Quotes (Knowledge Base) ---
# #             for q in data.get("text_quotes", []):
# #                 uid = f"{doc_name}_{q['quote_id']}"
# #                 if uid not in seen_quotes:
# #                     master_quotes.append({
# #                         "quote_id": uid,
# #                         "modality": "text",
# #                         "text": q["text"],
# #                         "image_path": "",
# #                         "doc_name": doc_name
# #                     })
# #                     seen_quotes.add(uid)
            
# #             # --- 3. Process Image Quotes (Knowledge Base) ---
# #             for img in data.get("img_quotes", []):
# #                 uid = f"{doc_name}_{img['quote_id']}"
# #                 if uid not in seen_quotes:
# #                     master_quotes.append({
# #                         "quote_id": uid,
# #                         "modality": "image",
# #                         "text": img["img_description"],
# #                         "image_path": img["img_path"], # Uses the real .jpg path from the file
# #                         "doc_name": doc_name
# #                     })
# #                     seen_quotes.add(uid)
            
# #             # --- 4. Save clean QA Pair for Evaluation ---
# #             # Handles q_id, id, or old_id
# #             query_id = data.get("q_id") if data.get("q_id") is not None else data.get("id", data.get("old_id", "0"))
            
# #             qa_pairs.append({
# #                 "q_id": query_id,
# #                 "question": clean_question,
# #                 "reference_answer": ref_answer,
# #                 "gold_quote_ids": [f"{doc_name}_{gid}" for gid in data.get("gold_quotes", [])]
# #             })

# # # Write to Files
# # with open(quotes_out, 'w', encoding='utf-8') as f:
# #     for q in master_quotes: f.write(json.dumps(q) + "\n")

# # with open(qa_out, 'w', encoding='utf-8') as f:
# #     for qa in qa_pairs: f.write(json.dumps(qa) + "\n")

# # print(f"✅ Success! Created {len(master_quotes)} unique quotes.")
# # print(f"✅ Created {len(qa_pairs)} clean QA pairs.")

# import json
# import os

# input_files = ["data/raw/train.jsonl", "data/raw/dev_15.jsonl", "data/raw/dev_20.jsonl", "data/raw/eval_15.jsonl", "data/raw/eval_20.jsonl"]
# quotes_out = "data/raw/quotes_master.jsonl"
# qa_out = "data/raw/qa_pairs.jsonl"

# seen_quotes = set()
# master_quotes = []
# qa_pairs = []

# # This will generate a guaranteed unique ID for every question
# qa_counter = 1 

# print("🚀 Starting Final MMDocRAG Refactoring (with Custom IDs)...")

# for file_name in input_files:
#     if not os.path.exists(file_name):
#         continue
        
#     with open(file_name, 'r', encoding='utf-8') as f:
#         for line in f:
#             data = json.loads(line)
#             doc_name = data.get("doc_name", "unknown")
            
#             # --- 1. Surgical Question Extraction ---
#             user_content = ""
#             ref_answer = ""
#             for msg in data.get("messages", []):
#                 if msg["role"] == "user":
#                     user_content = msg["content"]
#                 if msg["role"] == "assistant":
#                     ref_answer = msg["content"]

#             if "The user question is:" in user_content:
#                 clean_question = user_content.split("The user question is:")[-1].strip()
#             else:
#                 clean_question = user_content.split('\n')[-1].strip()

#             # --- 2. Process Text Quotes ---
#             for q in data.get("text_quotes", []):
#                 uid = f"{doc_name}_{q['quote_id']}"
#                 if uid not in seen_quotes:
#                     master_quotes.append({
#                         "quote_id": uid,
#                         "modality": "text",
#                         "text": q["text"],
#                         "image_path": "",
#                         "doc_name": doc_name
#                     })
#                     seen_quotes.add(uid)
            
#             # --- 3. Process Image Quotes ---
#             for img in data.get("img_quotes", []):
#                 uid = f"{doc_name}_{img['quote_id']}"
#                 if uid not in seen_quotes:
#                     master_quotes.append({
#                         "quote_id": uid,
#                         "modality": "image",
#                         "text": img["img_description"],
#                         "image_path": img["img_path"],
#                         "doc_name": doc_name
#                     })
#                     seen_quotes.add(uid)
            
#             # --- 4. Save clean QA Pair with OUR Custom ID ---
#             qa_pairs.append({
#                 "q_id": str(qa_counter),  # <--- FIXED: Now 1, 2, 3... 12220
#                 "question": clean_question,
#                 "reference_answer": ref_answer,
#                 "gold_quote_ids": [f"{doc_name}_{gid}" for gid in data.get("gold_quotes", [])]
#             })
            
#             qa_counter += 1  # Increment for the next question

# # Write to Files
# with open(quotes_out, 'w', encoding='utf-8') as f:
#     for q in master_quotes: f.write(json.dumps(q) + "\n")

# with open(qa_out, 'w', encoding='utf-8') as f:
#     for qa in qa_pairs: f.write(json.dumps(qa) + "\n")

# print(f"✅ Success! Created {len(master_quotes)} unique quotes.")
# print(f"✅ Created {len(qa_pairs)} clean QA pairs with unique IDs (1 to {qa_counter-1}).")
import json
import os
import random
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

INPUT_FILES = [
    "data/raw/train.jsonl",
    "data/raw/dev_15.jsonl",
    "data/raw/dev_20.jsonl",
    "data/raw/evaluation_15.jsonl",
    "data/raw/evaluation_20.jsonl",
]
TEST_OUT = "data/processed/test_set.jsonl"
SAMPLE_SIZE = 150
RANDOM_SEED = 42


def extract_qa(data: dict) -> tuple[str, str, str]:
    """
    Returns (clean_question, ref_answer, extraction_source).
    extraction_source is 'strategy_1', 'strategy_2', or 'fallback'.
    """
    clean_question = ""
    ref_answer = ""

    # --- STRATEGY 1: top-level keys (rows 4111+) ---
    q = data.get("question")
    a = data.get("answer_interleaved") or data.get("answer")

    if q:
        clean_question = (q or "").strip()
    if a:
        ref_answer = (a or "").strip()

    if clean_question and ref_answer:
        return clean_question, ref_answer, "strategy_1"

    # --- STRATEGY 2: messages array (rows 1-4110) ---
    user_content = ""
    for msg in data.get("messages", []):
        role = msg.get("role", "")
        content = msg.get("content", "") or ""
        if role == "user":
            user_content = content
        elif role == "assistant" and not ref_answer:
            ref_answer = content.strip()

    if not clean_question and user_content:
        if "The user question is:" in user_content:
            clean_question = user_content.split("The user question is:")[-1].strip()
        elif "User question is:" in user_content:
            clean_question = user_content.split("User question is:")[-1].strip()
        else:
            # Fragile fallback — last non-empty line
            lines = [l.strip() for l in user_content.splitlines() if l.strip()]
            clean_question = lines[-1] if lines else ""
            if clean_question:
                log.warning("Used last-line fallback to extract question: %r", clean_question[:80])

    if clean_question and ref_answer:
        return clean_question, ref_answer, "strategy_2"

    return clean_question, ref_answer, "fallback"


def main():
    log.info("Starting QA extraction...")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(TEST_OUT), exist_ok=True)

    qa_pairs = []
    qa_counter = 1
    skipped = 0

    for file_name in INPUT_FILES:
        if not os.path.exists(file_name):
            log.warning("File not found, skipping: %s", file_name)
            continue

        log.info("Processing: %s", file_name)

        with open(file_name, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue  # skip blank lines

                # Bug fix 1: handle malformed JSON gracefully
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    log.warning("Skipping malformed JSON in %s line %d: %s", file_name, line_num, e)
                    skipped += 1
                    continue

                doc_name = data.get("doc_name", "unknown")

                clean_question, ref_answer, source = extract_qa(data)

                # Bug fix 2: skip empty questions or answers
                if not clean_question or not ref_answer:
                    log.warning(
                        "Skipping record with missing %s in %s line %d (doc: %s)",
                        "question" if not clean_question else "answer",
                        file_name, line_num, doc_name
                    )
                    skipped += 1
                    continue

                qa_pairs.append({
                    "test_id": f"TEST_{qa_counter}",
                    "question": clean_question,
                    "reference_answer": ref_answer,
                    "gold_quote_ids": [
                        f"{doc_name}_{gid}" for gid in data.get("gold_quotes", [])
                    ],
                    "extraction_source": source,  # Tip: track which strategy fired
                    "source_file": file_name,
                })

                qa_counter += 1

    log.info("Extracted %d valid QA pairs (%d skipped).", len(qa_pairs), skipped)

    # Bug fix 3: guard against empty extraction
    if not qa_pairs:
        raise ValueError(
            "No QA pairs were extracted. Check that input files exist and are valid JSONL."
        )

    # Save the full extracted pool
    FULL_OUT = "data/processed/all_qa_pairs.jsonl"
    with open(FULL_OUT, "w", encoding="utf-8") as f:
        for qa in qa_pairs:
            f.write(json.dumps(qa) + "\n")
    log.info("Saved all %d QA pairs to %s", len(qa_pairs), FULL_OUT)

    # Sample — seed ensures reproducibility for a fixed pool
    random.seed(RANDOM_SEED)
    test_sample = random.sample(qa_pairs, min(SAMPLE_SIZE, len(qa_pairs)))

    with open(TEST_OUT, "w", encoding="utf-8") as f:
        for qa in test_sample:
            f.write(json.dumps(qa) + "\n")

    log.info(
        "Saved %d questions to %s  (pool size: %d, seed: %d)",
        len(test_sample), TEST_OUT, len(qa_pairs), RANDOM_SEED
    )

    # Strategy breakdown
    from collections import Counter
    sources = Counter(qa["extraction_source"] for qa in test_sample)
    for src, count in sorted(sources.items()):
        log.info("  %s: %d questions", src, count)


if __name__ == "__main__":
    main()