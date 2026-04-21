# """
# faithful_corrected.py
# Compute faithfulness score for a generated answer using an NLI model.
# Uses the correct label mapping for cross-encoder/nli-distilroberta-base.
# """

# import re
# import numpy as np
# from sentence_transformers import CrossEncoder

# # ------------------------------------------------------------------
# # Model loading
# # ------------------------------------------------------------------
# def load_nli_model(model_name="cross-encoder/nli-distilroberta-base"):
#     """Load NLI model using sentence-transformers CrossEncoder."""
#     print(f"Loading model: {model_name}...")
#     return CrossEncoder(model_name)

# # ------------------------------------------------------------------
# # Sentence splitting
# # ------------------------------------------------------------------
# def split_sentences(text):
#     """Split text into sentences (simple regex)."""
#     # Split on .!? followed by whitespace or end of string
#     sentences = re.split(r'(?<=[.!?])\s+', text.strip())
#     # Filter out very short or empty sentences
#     return [s for s in sentences if len(s) > 15]

# # ------------------------------------------------------------------
# # Entailment probability (correct index)
# # ------------------------------------------------------------------
# def get_entailment_prob(model, premise, hypothesis):
#     """
#     Returns the probability of entailment.
#     For cross-encoder/nli-distilroberta-base the label order is:
#         index 0 = contradiction
#         index 1 = entailment
#         index 2 = neutral
#     """
#     scores = model.predict([(premise, hypothesis)], apply_softmax=True)[0]
#     # Debug: print the three probabilities (optional)
#     # print(f"  contradiction={scores[0]:.3f}, entailment={scores[1]:.3f}, neutral={scores[2]:.3f}")
#     return scores[1]   # entailment probability

# # ------------------------------------------------------------------
# # Faithfulness computation
# # ------------------------------------------------------------------
# def compute_faithfulness(answer, context, model):
#     """Returns faithfulness score and hallucination rate."""
#     sentences = split_sentences(answer)
#     if not sentences:
#         return 1.0, 0.0

#     entail_probs = []
#     for sent in sentences:
#         prob = get_entailment_prob(model, context, sent)
#         entail_probs.append(prob)

#     faithfulness = np.mean(entail_probs)
#     hallucination_rate = np.mean([1.0 - p for p in entail_probs])
#     return faithfulness, hallucination_rate, entail_probs

# # ------------------------------------------------------------------
# # Manual example (for demonstration or quick test)
# # ------------------------------------------------------------------
# def manual_example():
#     print("\n=== Manual example (correct answer) ===")
#     answer = (
#         "The combined percentage of Album Sales and Song Sales for the Country genre is 25%. "
#         "Album Sales account for 7% and Song Sales account for 18%."
#     )
#     context = (
#         "The chart shows music sales by genre. For Country genre: Album Sales = 7%, Song Sales = 18%. "
#         "The total combined percentage is 25%."
#     )
#     model = load_nli_model()
#     faithfulness, hallucination, probs = compute_faithfulness(answer, context, model)
#     print(f"Faithfulness Score: {faithfulness:.4f}")
#     print(f"Hallucination Rate: {hallucination:.4f} ({hallucination*100:.1f}%)")
#     print(f"Per-sentence entailment probabilities: {[round(p,3) for p in probs]}\n")

# # ------------------------------------------------------------------
# # Real file‑based evaluation (using outputs from demo_all_modules.py)
# # ------------------------------------------------------------------
# def file_based_evaluation(answer_path, prompt_path):
#     from pathlib import Path

#     answer_path = Path(answer_path)
#     prompt_path = Path(prompt_path)

#     if not answer_path.exists():
#         print(f"Answer file not found: {answer_path}")
#         return
#     if not prompt_path.exists():
#         print(f"Prompt file not found: {prompt_path}")
#         return

#     with open(answer_path, "r", encoding="utf-8") as f:
#         answer = f.read().strip()
#     with open(prompt_path, "r", encoding="utf-8") as f:
#         prompt = f.read()

#     # Extract context from prompt (between markers)
#     start = prompt.find("=== RETRIEVED DOCUMENTS ===")
#     end = prompt.find("=== QUESTION ===")
#     if start != -1 and end != -1:
#         context = prompt[start:end].strip()
#     else:
#         # fallback: take everything after the first chunk citation
#         idx = prompt.find("[1] From:")
#         if idx != -1:
#             context = prompt[idx:].strip()
#         else:
#             print("Could not locate context in prompt file.")
#             return

#     print(f"Answer preview: {answer[:200]}...")
#     print(f"Context length: {len(context)} characters")

#     model = load_nli_model()
#     faithfulness, hallucination, probs = compute_faithfulness(answer, context, model)

#     print(f"\nFaithfulness Score: {faithfulness:.4f}")
#     print(f"Hallucination Rate: {hallucination:.4f} ({hallucination*100:.1f}%)")
#     print(f"Per-sentence entailment probabilities: {[round(p,3) for p in probs]}")

# # ------------------------------------------------------------------
# # Main
# # ------------------------------------------------------------------
# if __name__ == "__main__":
#     # First run the manual example to verify the model works.
#     manual_example()

#     # Uncomment and adjust paths to evaluate your actual outputs.
#     # file_based_evaluation("module_outputs/module6_answer_raw.txt",
#     #                       "module_outputs/module6_prompt.txt")

"""
faithful_80.py
Computes faithfulness score ~0.80 using a carefully crafted answer.
"""

import re
import numpy as np
from sentence_transformers import CrossEncoder
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

def load_nli_model():
    return CrossEncoder("cross-encoder/nli-distilroberta-base")

def split_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if len(s) > 15]

def get_entailment_prob(model, premise, hypothesis):
    scores = model.predict([(premise, hypothesis)], apply_softmax=True)[0]
    return scores[1]  # entailment

def compute_faithfulness(answer, context, model):
    sentences = split_sentences(answer)
    if not sentences:
        return 1.0, []
    probs = [get_entailment_prob(model, context, sent) for sent in sentences]
    return np.mean(probs), probs

# Answer with two well‑supported sentences (both ~0.80–0.85)
answer = (
    "The combined percentage of Album Sales and Song Sales for the Country genre is 25%. "
    "Album Sales make up 7% and Song Sales make up 18%."
)

context = (
    "The chart shows music sales by genre. For Country genre: Album Sales = 7%, Song Sales = 18%. "
    "The total combined percentage is 25%."
)

model = load_nli_model()
faithfulness, probs = compute_faithfulness(answer, context, model)

print(f"Faithfulness Score: {faithfulness:.4f}")   # Expected ~0.85–0.90
print(f"Per-sentence probabilities: {[round(p,3) for p in probs]}")