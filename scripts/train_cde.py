#train_cde.py

import json
import logging
import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_PATH = "data/raw/train.jsonl"

label_map = {
    "text": 0,
    "figure": 1,
    "table": 2,
    "hybrid": 3
}

def build_dataset():
    logger.info("Parsing dataset and mapping labels...")
    questions = []
    labels = []

    try:
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    row = json.loads(line)
                    print(json.dumps(row["messages"], indent=2))
                    break # Just print the first one to verify the structure
                    q = row.get("question")
                    
                    if not q:
                        continue # Skip malformed rows

                    modality = row.get("evidence_modality_type", [])

                    # Routing logic
                    if len(modality) == 1:
                        label = modality[0]
                    else:
                        label = "hybrid"

                    # Fallback for unrecognized labels
                    if label not in label_map:
                        label = "text"

                    questions.append(q)
                    labels.append(label_map[label])
                except json.JSONDecodeError:
                    continue
                    
    except FileNotFoundError:
        logger.error(f"Could not find training data at {DATA_PATH}")
        raise

    logger.info(f"Successfully loaded {len(questions)} training examples.")
    return Dataset.from_dict({
        "text": questions,
        "label": labels
    })

def compute_metrics(eval_pred):
    """Calculates accuracy and F1 score for your thesis report."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    
    return {
        "accuracy": acc,
        "f1": f1
    }

def main():
    dataset = build_dataset()
    dataset = dataset.train_test_split(test_size=0.1)

    logger.info("Loading tokenizer...")
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            # Let the collator handle dynamic padding for efficiency
            max_length=64 
        )

    logger.info("Tokenizing dataset...")
    dataset = dataset.map(tokenize, batched=True)
    
    # Dynamically pads batches to the longest sequence in that specific batch (saves memory)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    logger.info("Loading DistilBERT model...")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=4
    )

    args = TrainingArguments(
    output_dir="models/cde",
    learning_rate=3e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    fp16=torch.cuda.is_available(),
    logging_dir="./logs",
    logging_steps=50,
    report_to="none"
    )   

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics # Now you get real stats!
    )

    logger.info("Starting training loop...")
    trainer.train()

    logger.info("Training complete. Saving model to disk...")
    trainer.save_model("models/cde")
    logger.info("Context Decision Engine successfully saved!")

if __name__ == "__main__":
    main()