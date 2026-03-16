"""
train_context_validator.py
---------------------------
Trains the Context Validation Model from your architecture diagram.

Model: DistilBERT fine-tuned as a 3-class classifier
  Input:  [CLS] question [SEP] chunk_text [SEP]
  Output: label 0 (irrelevant) | 1 (partial) | 2 (highly relevant)

Training data: data/processed/validation_pairs_train.jsonl
Val data:      data/processed/validation_pairs_dev.jsonl
Output:        models/context_validator/

Fixes in this version:
  - tqdm progress bar so you can see training is running
  - Dev file fallback: if validation_pairs_dev.jsonl is empty,
    automatically splits 10% of train data for validation
  - Logs loss every 100 steps so you know it's working
  - num_workers=0 on Windows to avoid DataLoader deadlock

Usage:
    python scripts/train_context_validator.py --epochs 3 --batch_size 32

Quick test (5000 samples, 1 epoch):
    python scripts/train_context_validator.py --epochs 1 --batch_size 32 --max_train 5000
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR   = Path("data/processed")
MODELS_DIR = Path("models/context_validator")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Windows fix: DataLoader num_workers must be 0 on Windows or causes deadlock
import platform
NUM_WORKERS = 0 if platform.system() == "Windows" else 2


# ------------------------------------------------------------------ #
# Dataset
# ------------------------------------------------------------------ #

class ValidationPairDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 256):
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        self.samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        self.tokenizer  = tokenizer
        self.max_length = max_length
        log.info("Loaded %d samples from %s", len(self.samples), jsonl_path)

        # Log label distribution
        labels = [s["label"] for s in self.samples]
        for lbl in [0, 1, 2]:
            count = labels.count(lbl)
            log.info("  Label %d: %d samples (%.1f%%)", lbl, count, 100*count/max(len(labels),1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample   = self.samples[idx]
        question = sample["question"]
        chunk    = sample["chunk"]
        label    = sample["label"]

        encoding = self.tokenizer(
            question,
            chunk,
            max_length=self.max_length,
            truncation="only_second",
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label":          torch.tensor(label, dtype=torch.long),
        }


# ------------------------------------------------------------------ #
# Training loop with tqdm progress bar
# ------------------------------------------------------------------ #

def train_epoch(model, loader, optimizer, scheduler, device, epoch, total_epochs):
    model.train()
    total_loss  = 0
    log_every   = max(1, len(loader) // 10)  # log 10 times per epoch

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [train]", leave=True)
    for step, batch in enumerate(pbar, 1):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss    = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        avg_loss    = total_loss / step

        # Update progress bar with live loss
        pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

    return total_loss / len(loader)


def evaluate(model, loader, device, desc="val") -> dict:
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0

    pbar = tqdm(loader, desc=f"  [{desc}]", leave=False)
    with torch.no_grad():
        for batch in pbar:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            preds   = outputs.logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total_loss += outputs.loss.item()

    # Fix: only report labels that actually appear — label 0 never exists
    # in this dataset, so 3 target_names with 2 present classes crashed
    present_labels = sorted(set(all_labels) | set(all_preds))
    label_name_map = {0: "irrelevant", 1: "partial", 2: "relevant"}
    target_names   = [label_name_map[l] for l in present_labels]

    f1     = f1_score(all_labels, all_preds, average="macro",
                      labels=present_labels, zero_division=0)
    report = classification_report(
        all_labels, all_preds,
        labels=present_labels,
        target_names=target_names,
        zero_division=0,
    )
    return {
        "loss":   total_loss / len(loader),
        "f1":     f1,
        "report": report,
    }


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="distilbert-base-uncased")
    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=2e-5)
    parser.add_argument("--max_length", type=int,   default=256)
    parser.add_argument("--max_train",  type=int,   default=None,
                        help="Limit samples for quick testing (e.g. --max_train 5000)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s  |  Model: %s  |  Epochs: %d  |  Batch: %d",
             device, args.model, args.epochs, args.batch_size)

    # Load tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model     = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=3
    ).to(device)

    # ── Load datasets ───────────────────────────────────────────────
    train_path = DATA_DIR / "validation_pairs_train.jsonl"
    dev_path   = DATA_DIR / "validation_pairs_dev.jsonl"

    if not train_path.exists():
        log.error("Training data not found: %s", train_path)
        log.error("Run prepare_validation_data.py first.")
        sys.exit(1)

    full_ds = ValidationPairDataset(str(train_path), tokenizer, args.max_length)

    # Fix: if dev file is empty or missing, auto-split 10% from train
    use_dev_split = False
    if dev_path.exists():
        dev_ds = ValidationPairDataset(str(dev_path), tokenizer, args.max_length)
        if len(dev_ds) == 0:
            log.warning("Dev file exists but has 0 samples — auto-splitting 10%% of train for validation.")
            use_dev_split = True
    else:
        log.warning("Dev file not found — auto-splitting 10%% of train for validation.")
        use_dev_split = True

    if use_dev_split:
        val_size   = int(0.10 * len(full_ds))
        train_size = len(full_ds) - val_size
        train_ds, dev_ds = random_split(
            full_ds, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        log.info("Auto-split: %d train / %d val", len(train_ds), len(dev_ds))
    else:
        train_ds = full_ds

    # Optional subsample for quick testing
    if args.max_train and len(train_ds) > args.max_train:
        indices  = np.random.choice(len(train_ds), args.max_train, replace=False)
        train_ds = Subset(train_ds, indices)
        log.info("Subsampled training set to %d samples", len(train_ds))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"))
    dev_loader   = DataLoader(dev_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"))

    log.info("Train batches: %d  |  Val batches: %d", len(train_loader), len(dev_loader))

    # ── Optimizer + scheduler ───────────────────────────────────────
    optimizer    = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps  = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    best_f1   = 0.0
    best_path = MODELS_DIR / "best_model"

    log.info("Training started — watch the progress bar below.")
    log.info("Expected time: ~15-25 min/epoch on GPU, ~60+ min/epoch on CPU.")

    # ── Training loop ───────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler,
                                 device, epoch, args.epochs)
        metrics    = evaluate(model, dev_loader, device)

        log.info(
            "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_f1=%.4f",
            epoch, args.epochs, train_loss, metrics["loss"], metrics["f1"]
        )
        print(metrics["report"])

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            model.save_pretrained(str(best_path))
            tokenizer.save_pretrained(str(best_path))
            log.info("  ✓ New best model saved (f1=%.4f) → %s", best_f1, best_path)

    log.info("Training complete. Best val F1: %.4f", best_f1)
    log.info("Model saved to: %s", best_path.resolve())

    # Save config
    cfg = {
        "base_model":  args.model,
        "num_labels":  3,
        "max_length":  args.max_length,
        "best_val_f1": round(best_f1, 4),
        "label_map":   {"0": "irrelevant", "1": "partial", "2": "relevant"},
    }
    with open(MODELS_DIR / "training_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    log.info("Config saved to %s", MODELS_DIR / "training_config.json")


if __name__ == "__main__":
    main()