"""
prompt_builder.py
-----------------
Builds prompts for Qwen2.5-VL from retrieved RAG context.
Works for ANY document type — financial, academic, research, school data, etc.

Functions:
    build_text_only_prompt(query, text_hits, image_hits)  -> (user_text, [])
    build_multimodal_prompt(query, text_hits, image_hits) -> (user_text, image_paths)
    SYSTEM_PROMPT  — shared system instruction string
"""

SYSTEM_PROMPT = (
    "You are a knowledgeable assistant. Answer questions using ONLY the "
    "retrieved document context provided below.\n\n"
    "CRITICAL RULES:\n"
    "1. ABBREVIATIONS IN INDIAN CONTEXT: SC = Scheduled Caste, ST = Scheduled Tribe, "
    "BC = Backward Class, MBC = Most Backward Class, OBC = Other Backward Class, "
    "TNEA = Tamil Nadu Engineering Admissions. Never interpret SC as a country code.\n"
    "2. Read every text passage and image carefully before answering.\n"
    "3. Extract exact numbers, policy details, or eligibility criteria from context.\n"
    "4. State which document the answer came from.\n"
    "5. Only say not found if the information is genuinely absent from all context.\n"
    "6. Give direct, specific answers — state the fact first."
)


# ------------------------------------------------------------------ #
# Context formatters
# ------------------------------------------------------------------ #

def _fmt_text(text_hits: list[dict]) -> str:
    if not text_hits:
        return ""
    lines = ["=== RETRIEVED TEXT PASSAGES ==="]
    for i, h in enumerate(text_hits, 1):
        doc  = h.get("doc_name", "unknown")
        text = (h.get("text") or "").strip()[:600]
        page = h.get("page_id")
        page_str = f" | Page {page}" if page else ""
        lines.append(f"\n[Passage {i} | {doc}{page_str}]\n{text}")
    return "\n".join(lines)


def _fmt_image_desc(image_hits: list[dict]) -> str:
    if not image_hits:
        return ""
    lines = ["=== RETRIEVED IMAGES (charts / tables / figures) ==="]
    for i, h in enumerate(image_hits, 1):
        doc  = h.get("doc_name", "unknown")
        path = h.get("image_path", "")
        desc = (h.get("text") or "").strip()[:400]
        page = h.get("page_id")
        page_str = f" | Page {page}" if page else ""
        cde  = h.get("cde_confidence")
        cde_str = f" [CDE: {cde:.3f}]" if cde else ""
        lines.append(f"\n[Image {i} | {doc}{page_str}{cde_str}]\n{desc}")
    return "\n".join(lines)


# ------------------------------------------------------------------ #
# Prompt builders
# ------------------------------------------------------------------ #

def build_text_only_prompt(
    query:      str,
    text_hits:  list[dict],
    image_hits: list[dict] = None,
) -> tuple[str, list]:
    """Text-only — image captions included as text but no actual images sent."""
    parts = []
    txt = _fmt_text(text_hits)
    if txt:
        parts.append(txt)
    img = _fmt_image_desc(image_hits or [])
    if img:
        parts.append(img)
    parts.append(f"\n=== QUESTION ===\n{query}")
    return "\n\n".join(parts), []


def build_multimodal_prompt(
    query:      str,
    text_hits:  list[dict],
    image_hits: list[dict] = None,
    max_images: int = 3,
) -> tuple[str, list]:
    """
    Multimodal — returns (user_text, image_paths).
    Qwen2.5-VL reads the actual image files alongside text context.
    """
    image_hits = image_hits or []
    parts = []

    txt = _fmt_text(text_hits)
    if txt:
        parts.append(txt)

    img_desc = _fmt_image_desc(image_hits)
    if img_desc:
        parts.append(img_desc)
        parts.append(
            "The images above are attached. Examine them carefully — "
            "they may contain charts, tables, or figures with the exact "
            "values needed to answer the question."
        )

    parts.append(f"\n=== QUESTION ===\n{query}")
    user_text   = "\n\n".join(parts)
    image_paths = [h["image_path"] for h in image_hits[:max_images]
                   if h.get("image_path")]
    return user_text, image_paths


def format_answer(
    query:      str,
    answer:     str,
    text_hits:  list[dict],
    image_hits: list[dict],
    arch_name:  str,
    mode:       str,
) -> dict:
    """Package result into a structured dict for saving."""
    return {
        "query":         query,
        "answer":        answer,
        "arch_name":     arch_name,
        "mode":          mode,
        "sources_text":  [{"doc": h.get("doc_name"), "quote_id": h.get("quote_id"),
                           "page": h.get("page_id")} for h in text_hits],
        "sources_image": [{"doc": h.get("doc_name"), "path": h.get("image_path"),
                           "page": h.get("page_id")} for h in image_hits],
    }