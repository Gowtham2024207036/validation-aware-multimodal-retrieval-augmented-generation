"""
compute_faithfulness.py
Compute faithfulness score for a generated answer using an NLI model.
Uses sentence_transformers.CrossEncoder to avoid tokenizer issues.
"""

import os
import re
from pathlib import Path
import numpy as np
from sentence_transformers import CrossEncoder

# Paths (adjust if your outputs are in a different folder)
OUTPUT_DIR = Path("module_outputs")
ANSWER_PATH = OUTPUT_DIR / "module6_answer_raw.txt"
PROMPT_PATH = OUTPUT_DIR / "module6_prompt.txt"

def load_nli_model(model_name="cross-encoder/nli-distilroberta-base"):
    """Load NLI model using CrossEncoder."""
    try:
        return CrossEncoder(model_name)
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        print("Trying fallback model cross-encoder/nli-deberta-v3-base...")
        return CrossEncoder("cross-encoder/nli-deberta-v3-base")

def split_sentences(text):
    """Split text into sentences using regex."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    # Filter out very short sentences (likely incomplete)
    return [s for s in sentences if len(s) > 15]

def get_entailment_prob(model, premise, hypothesis):
    """Return probability of entailment (class 2)."""
    scores = model.predict([(premise, hypothesis)], apply_softmax=True)[0]
    # Order: [contradiction, neutral, entailment]
    return scores[2]

def compute_faithfulness(answer, context, model):
    """Compute faithfulness score and hallucination rate."""
    sentences = split_sentences(answer)
    if not sentences:
        return 1.0, 0.0, []

    entail_scores = []
    for sent in sentences:
        prob = get_entailment_prob(model, context, sent)
        entail_scores.append(prob)
    faithfulness = np.mean(entail_scores)
    hallucination_rate = sum(1 for p in entail_scores if p < 0.5) / len(sentences)
    return faithfulness, hallucination_rate, entail_scores

def extract_context_from_prompt(prompt_text):
    """Extract the retrieved context from the prompt."""
    # First try standard markers
    start = prompt_text.find("=== RETRIEVED DOCUMENTS ===")
    end = prompt_text.find("=== QUESTION ===")
    if start != -1 and end != -1:
        return prompt_text[start:end].strip()

    # Fallback: find first chunk citation marker
    idx = prompt_text.find("[1] From:")
    if idx != -1:
        return prompt_text[idx:].strip()

    # Last resort: whole prompt (will likely give low score)
    print("⚠️  Could not locate context markers. Using whole prompt as context.")
    return prompt_text

def main():
    # Check files exist
    if not ANSWER_PATH.exists():
        print(f"Answer file not found: {ANSWER_PATH}")
        print("Run demo_all_modules.py first.")
        return
    if not PROMPT_PATH.exists():
        print(f"Prompt file not found: {PROMPT_PATH}")
        return

    with open(ANSWER_PATH, "r", encoding="utf-8") as f:
        answer = f.read().strip()
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        prompt = f.read()

    print("=== Answer (first 300 chars) ===")
    print(answer[:300])
    print("\n=== Prompt (first 300 chars) ===")
    print(prompt[:300])

    context = extract_context_from_prompt(prompt)
    print(f"\n=== Extracted context length: {len(context)} characters ===")
    if len(context) < 100:
        print("⚠️  Context is very short. This may cause low faithfulness scores.")
        print("Check that the prompt contains the retrieved chunks.")
    else:
        print("Context preview (first 300 chars):")
        print(context[:300])

    if not context:
        print("Error: No context extracted. Cannot compute faithfulness.")
        return

    print("\nLoading NLI model...")
    model = load_nli_model()
    print("Computing faithfulness...")
    faithfulness, hallucination_rate, entail_scores = compute_faithfulness(answer, context, model)

    print(f"\n=== RESULTS ===")
    print(f"Number of sentences: {len(split_sentences(answer))}")
    print(f"Faithfulness Score: {faithfulness:.4f}")
    print(f"Hallucination Rate: {hallucination_rate:.4f} ({hallucination_rate*100:.1f}%)")
    print(f"Per‑sentence entailment probabilities: {[round(p,3) for p in entail_scores]}")

if __name__ == "__main__":
    main()