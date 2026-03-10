import json

input_file = "data/processed/qa_pairs.jsonl"
output_file = "data/processed/modality_train.jsonl"

image_keywords = ["figure", "diagram", "chart", "graph", "plot", "image"]
table_keywords = ["table", "accuracy", "percentage", "rate", "comparison"]

def detect_modality(question):

    q = question.lower()

    for k in image_keywords:
        if k in q:
            return "image"

    for k in table_keywords:
        if k in q:
            return "table"

    return "text"


with open(input_file, "r", encoding="utf8") as f, \
     open(output_file, "w", encoding="utf8") as out:

    for line in f:

        data = json.loads(line)

        question = data["question"]

        label = detect_modality(question)

        out.write(json.dumps({
            "text": question,
            "label": label
        }) + "\n")

print("Dataset created:", output_file)