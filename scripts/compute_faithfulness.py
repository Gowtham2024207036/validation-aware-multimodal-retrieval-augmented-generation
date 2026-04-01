"""
compute_faithfulness.py
Computes faithfulness score for the generated answer using an NLI model.
Requires that demo_all_modules.py has been run.
"""

import os
import sys
import re
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR = Path("module_outputs")
def load_nli_model():
    model_name = "cross-encoder/nli-distilroberta-base"  # smaller and more stable
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

def split_into_claims(text):
    """Simple sentence splitting (more sophisticated: use spaCy or nltk)."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]

def entailment_prob(premise, hypothesis, tokenizer, model):
    """Return probability of entailment (label 0 = contradiction, 1 = neutral, 2 = entailment)."""
    inputs = tokenizer(premise, hypothesis, truncation=True, padding=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        # probs[2] is entailment probability
        return probs[2]

def main():
    # Read the raw answer (module 6 output)
    answer_path = OUTPUT_DIR / "module6_answer_raw.txt"
    if not answer_path.exists():
        print("No raw answer found. Run demo_all_modules.py first.")
        return
    with open(answer_path, "r", encoding="utf-8") as f:
        answer = f.read().strip()
    if not answer or "LM Studio offline" in answer:
        print("Answer not available or LM Studio was offline.")
        return

    # Read the context from the prompt (module 6 prompt)
    prompt_path = OUTPUT_DIR / "module6_prompt.txt"
    if not prompt_path.exists():
        print("No prompt found. Can't compute faithfulness without context.")
        return
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt = f.read()
    # Extract the context part (everything between "=== RETRIEVED DOCUMENTS ===" and "=== QUESTION ===")
    context = ""
    if "=== RETRIEVED DOCUMENTS ===" in prompt and "=== QUESTION ===" in prompt:
        start = prompt.find("=== RETRIEVED DOCUMENTS ===")
        end = prompt.find("=== QUESTION ===")
        context = prompt[start:end].strip()
    else:
        print("Could not extract context from prompt. Using full prompt as context.")
        context = prompt

    # Load NLI model
    print("Loading NLI model...")
    tokenizer, model = load_nli_model()

    # Split answer into claims
    claims = split_into_claims(answer)
    if not claims:
        print("No claims extracted.")
        return

    print(f"Answer has {len(claims)} claims.")
    entail_scores = []
    for i, claim in enumerate(claims, 1):
        prob = entailment_prob(context, claim, tokenizer, model)
        entail_scores.append(prob)
        print(f"Claim {i}: {claim[:80]}...")
        print(f"  Entailment probability: {prob:.3f}")

    # Faithfulness = average entailment probability (or could be thresholded)
    faithfulness = np.mean(entail_scores)
    print(f"\nFaithfulness Score: {faithfulness:.3f}")

    # Optionally also compute citation coverage
    # Extract citations from answer (like [doc, page])
    citations = re.findall(r'\[(.*?)(?:, Page (\d+))?\]', answer)
    if citations:
        # Check if each cited chunk exists in the context (simple string match)
        existing = sum(1 for doc, page in citations if doc in context)
        citation_coverage = existing / len(citations) if citations else 1.0
        print(f"Citation coverage: {citation_coverage:.3f} ({existing}/{len(citations)})")
        # Combine scores (e.g., average)
        faithfulness = (faithfulness + citation_coverage) / 2
        print(f"Combined Faithfulness (incl. citations): {faithfulness:.3f}")

if __name__ == "__main__":
    main()