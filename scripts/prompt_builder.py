"""
prompt_builder.py
-----------------
Builds structured prompts for Qwen2.5-VL from retrieved RAG context.

Three prompt modes:
  1. text_only   — only text chunks passed as context
  2. image_only  — only image descriptions + base64 images passed
  3. multimodal  — text chunks + images combined (best for financial QA)

System prompt is tuned for financial document QA:
  - Instructs model to answer only from provided context
  - Asks for specific numbers/figures when available
  - Instructs it to say "not found" rather than hallucinate
"""

from lmstudio_client import build_multimodal_message

SYSTEM_PROMPT = """You are a financial analyst assistant. You answer questions about company financials strictly based on the provided context — retrieved text passages and financial charts/tables from annual reports (10-K filings).

Rules:
1. Answer ONLY using the provided context. Do not use external knowledge.
2. If the exact answer is in the context, quote the specific number or figure.
3. If the answer cannot be found in the context, say: "The information was not found in the retrieved documents."
4. When referencing figures, mention which document they came from (e.g. "According to COSTCO_2021_10K...").
5. Be concise — answer in 2-4 sentences unless a longer explanation is needed.
6. If financial images/tables are provided, analyse them carefully for relevant numbers."""


def format_text_context(text_hits: list[dict]) -> str:
    """Format retrieved text chunks into a readable context block."""
    if not text_hits:
        return "No text passages retrieved."
    lines = ["=== RETRIEVED TEXT PASSAGES ==="]
    for i, hit in enumerate(text_hits, 1):
        doc  = hit.get("doc_name", "unknown")
        text = (hit.get("text") or "").strip()
        lines.append(f"\n[Passage {i} — {doc}]\n{text}")
    return "\n".join(lines)


def format_image_context(image_hits: list[dict]) -> str:
    """Format image descriptions (captions) as text context."""
    if not image_hits:
        return "No financial images retrieved."
    lines = ["=== RETRIEVED FINANCIAL IMAGES (descriptions) ==="]
    for i, hit in enumerate(image_hits, 1):
        doc  = hit.get("doc_name", "unknown")
        path = hit.get("image_path", "")
        desc = (hit.get("text") or "").strip()
        lines.append(f"\n[Image {i} — {doc} | {path}]\nDescription: {desc}")
    return "\n".join(lines)


def build_text_only_prompt(query: str, text_hits: list[dict]) -> list[dict]:
    """Prompt using only retrieved text chunks."""
    context = format_text_context(text_hits)
    user_msg = f"{context}\n\n=== QUESTION ===\n{query}"
    return [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": user_msg},
    ]


def build_image_only_prompt(
    query: str,
    image_hits: list[dict],
) -> list[dict]:
    """
    Prompt using only retrieved images (passed as base64 to Qwen2.5-VL).
    The model directly reads the financial tables/charts from the images.
    """
    if not image_hits:
        return build_text_only_prompt(query, [])

    image_paths = [h["image_path"] for h in image_hits if h.get("image_path")]
    text_prompt = (
        f"The following financial chart(s) or table(s) have been retrieved "
        f"from annual reports to help answer this question.\n\n"
        f"Question: {query}\n\n"
        f"Please analyse the image(s) carefully and answer based on what you see."
    )
    user_msg = build_multimodal_message(text_prompt, image_paths)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        user_msg,
    ]


def build_multimodal_prompt(
    query: str,
    text_hits: list[dict],
    image_hits: list[dict],
    max_images: int = 2,
) -> list[dict]:
    """
    Full multimodal prompt: text chunks as string context + images as base64.
    This is the most powerful mode — Qwen2.5-VL reads both simultaneously.

    Note: We cap at max_images=2 by default to keep the prompt within
    Qwen2.5-VL-7B's context window. Increase if you have enough VRAM.
    """
    text_context = format_text_context(text_hits)
    img_desc     = format_image_context(image_hits)

    text_prompt = (
        f"{text_context}\n\n"
        f"{img_desc}\n\n"
        f"The images above are financial charts/tables from the same documents. "
        f"Use both the text passages AND the visual content of the images to answer.\n\n"
        f"=== QUESTION ===\n{query}"
    )

    image_paths = [h["image_path"] for h in image_hits if h.get("image_path")]
    user_msg    = build_multimodal_message(text_prompt, image_paths, max_images=max_images)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        user_msg,
    ]
