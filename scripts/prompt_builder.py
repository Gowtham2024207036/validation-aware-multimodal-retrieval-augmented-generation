"""
prompt_builder.py - Universal prompt builder
No hardcoded keywords - works with ANY document type
"""

def build_prompt(query, text_hits, image_hits=None):
    """Build prompt dynamically from whatever was retrieved"""
    parts = []
    
    if text_hits:
        parts.append("=== DOCUMENTS ===")
        for i, h in enumerate(text_hits[:5], 1):  # Top 5 chunks
            doc = h.get("doc_name", "Unknown").replace("_", " ").title()
            text = h.get("text", "").strip()
            page = h.get("page_id", "")
            page_str = f" [Page {page}]" if page else ""
            parts.append(f"\n[{i}] From: {doc}{page_str}\n{text[:1000]}")
    
    if image_hits:
        parts.append("\n=== IMAGES ===")
        for i, h in enumerate(image_hits[:3], 1):
            doc = h.get("doc_name", "Unknown").replace("_", " ").title()
            desc = h.get("text", "").strip()
            page = h.get("page_id", "")
            page_str = f" [Page {page}]" if page else ""
            parts.append(f"\n[Image {i}] From: {doc}{page_str}\nDescription: {desc}")
    
    parts.append(f"\n=== QUESTION ===\n{query}")
    parts.append("\n=== ANSWER ===")
    
    return "\n\n".join(parts)

# Universal system prompt - works for ANY document type
SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question using ONLY the provided documents and images. "
    "If the answer is in the documents, provide specific details with citations. "
    "If the answer is not found in any document, say 'The answer is not in the uploaded documents.' "
    "Always cite which document and page your information comes from using [Document Name, Page X] format."
)