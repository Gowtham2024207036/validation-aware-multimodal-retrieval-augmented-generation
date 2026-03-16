# import json
# import os

# # Files to process
# input_files = ["data/raw/train.jsonl", "data/raw/dev_20.jsonl", "data/raw/eval_20.jsonl"]
# quotes_out = "data/processed/quotes_master_final.jsonl"
# qa_out = "data/processed/qa_pairs_final.jsonl"

# seen_quotes = set()
# master_quotes = []
# qa_pairs = []

# print("🚀 Starting MMDocRAG Data Refactoring...")

# for file_name in input_files:
#     if not os.path.exists(file_name):
#         continue
        
#     with open(file_name, 'r', encoding='utf-8') as f:
#         for line in f:
#             data = json.loads(line)
#             doc_name = data.get("doc_name", "unknown")
            
#             # 1. Process Text Quotes
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
            
#             # 2. Process Image/Table Quotes
#             for img in data.get("img_quotes", []):
#                 uid = f"{doc_name}_{img['quote_id']}"
#                 if uid not in seen_quotes:
#                     master_quotes.append({
#                         "quote_id": uid,
#                         "modality": "image",
#                         "text": img["img_description"], # Description for CLIP
#                         "image_path": img["img_path"],   # Path to the .jpg
#                         "doc_name": doc_name
#                     })
#                     seen_quotes.add(uid)
            
#             # 3. Create Clean QA Pair for Testing
#             query_id = data.get("q_id") or data.get("id") or data.get("old_id") or 0
            
#             qa_pairs.append({
#                 "q_id": query_id,
#                 "question": data.get("question", ""),
#                 "reference_answer": data.get("answer_interleaved", ""),
#                 "gold_quote_ids": [f"{doc_name}_{gid}" for gid in data.get("gold_quotes", [])]
#             })

# # Save Master Quotes (For Indexing)
# with open(quotes_out, 'w', encoding='utf-8') as f:
#     for q in master_quotes:
#         f.write(json.dumps(q) + "\n")

# # Save QA Pairs (For Testing)
# with open(qa_out, 'w', encoding='utf-8') as f:
#     for qa in qa_pairs:
#         f.write(json.dumps(qa) + "\n")

# print(f"✅ Success! Created {len(master_quotes)} unique quotes and {len(qa_pairs)} QA pairs.")

import json
import os
import re

input_files = ["data/raw/train.jsonl", "data/raw/dev_15.jsonl", "data/raw/dev_20.jsonl", "data/raw/eval_15.jsonl", "data/raw/eval_20.jsonl"]
quotes_out = "data/processed/quotes_master.jsonl"
qa_out = "data/processed/qa_pairs.jsonl"

seen_quotes = set()
master_quotes = []
qa_pairs = []

print("🚀 Starting Deep-Dive MMDocRAG Refactoring...")

for file_name in input_files:
    if not os.path.exists(file_name):
        continue
        
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            doc_name = data.get("doc_name", "unknown")
            messages = data.get("messages", [])
            
            # --- 1. Extract the Question and Reference Answer ---
            user_query = ""
            ref_answer = ""
            
            for msg in messages:
                if msg["role"] == "user":
                    # The question is usually at the end of the user content
                    content = msg["content"]
                    if "User question is:" in content:
                        user_query = content.split("User question is:")[-1].strip()
                    else:
                        user_query = content # Fallback
                        
                if msg["role"] == "assistant":
                    ref_answer = msg["content"]

            # --- 2. Process Text Quotes ---
            for q in data.get("text_quotes", []):
                uid = f"{doc_name}_{q['quote_id']}"
                if uid not in seen_quotes:
                    master_quotes.append({
                        "quote_id": uid,
                        "modality": "text",
                        "text": q["text"],
                        "image_path": "",
                        "doc_name": doc_name
                    })
                    seen_quotes.add(uid)
            
            # --- 3. Process Image Quotes ---
            for img in data.get("img_quotes", []):
                uid = f"{doc_name}_{img['quote_id']}"
                if uid not in seen_quotes:
                    master_quotes.append({
                        "quote_id": uid,
                        "modality": "image",
                        "text": img["img_description"],
                        "image_path": img["img_path"],
                        "doc_name": doc_name
                    })
                    seen_quotes.add(uid)
            
            # --- 4. Save the QA Pair ---
            query_id = data.get("q_id") or data.get("id") or data.get("old_id")
            qa_pairs.append({
                "q_id": query_id,
                "question": user_query,
                "reference_answer": ref_answer,
                "gold_quote_ids": [f"{doc_name}_{gid}" for gid in data.get("gold_quotes", [])]
            })

# Write files
with open(quotes_out, 'w', encoding='utf-8') as f:
    for q in master_quotes: f.write(json.dumps(q) + "\n")

with open(qa_out, 'w', encoding='utf-8') as f:
    for qa in qa_pairs: f.write(json.dumps(qa) + "\n")

print(f"✅ Success! Created {len(master_quotes)} unique quotes and {len(qa_pairs)} QA pairs with actual text!")