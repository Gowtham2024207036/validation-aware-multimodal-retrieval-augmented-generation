"""
app.py - FAST Document QA System for Demo
Optimized for speed - answers in seconds, not minutes
"""

import os
import sys
import threading
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
from PIL import Image

from base_retriever import SharedModels
from lmstudio_client import check_connection, generate_text_only, LM_STUDIO_BASE_URL
import arch6_full_proposed as best_arch

# Import ingest
try:
    from document_ingestor import ingest_pdf, ingest_image
except ImportError:
    def ingest_pdf(*args, **kwargs): return False, "Document ingestor not available"
    def ingest_image(*args, **kwargs): return {"success": False, "message": "Not available"}

import config

IMAGES_ROOT = Path(config.RAW_DATA_DIR)
_models = None
_lock = threading.Lock()

def get_models():
    global _models
    with _lock:
        if _models is None:
            try:
                _models = SharedModels()
                print("✅ Models loaded")
            except Exception as e:
                print(f"❌ Error: {e}")
                return None
    return _models

def upload_pdf(file_obj, doc_name_input):
    """Fast PDF upload"""
    if file_obj is None:
        return "❌ No file selected", None
    
    try:
        uploaded_path = file_obj.name
        doc_name = doc_name_input.strip() or Path(uploaded_path).stem.replace(" ", "_").upper()
        
        success, message = ingest_pdf(uploaded_path, doc_name)
        
        if success:
            from base_retriever import build_bm25_index
            models = get_models()
            if models:
                models.text_bm25, models.text_records = build_bm25_index(
                    models.client, config.TEXT_COLLECTION
                )
                models.image_bm25, models.image_records = build_bm25_index(
                    models.client, config.IMAGE_COLLECTION
                )
            return f"✅ {message}", None
        else:
            return f"❌ {message}", None
    except Exception as e:
        return f"❌ Error: {str(e)}", None

def upload_image(pil_img, doc_name_input):
    """Fast image upload"""
    if pil_img is None:
        return "❌ No image selected", None
    
    doc_name = doc_name_input.strip() or "UPLOADED_IMAGE"
    
    try:
        if not isinstance(pil_img, Image.Image):
            pil_img = Image.fromarray(pil_img)
        
        result = ingest_image(pil_img, doc_name, page_id=1)
        
        if result.get("success"):
            return f"✅ {result['message']}", None
        else:
            return f"❌ {result['message']}", None
    except Exception as e:
        return f"❌ Error: {str(e)}", None

def run_pipeline(query):
    """FAST pipeline - answers in seconds"""
    if not query.strip():
        return "Please enter a question.", [], "No sources cited"
    
    # Quick LM Studio check
    try:
        import requests
        requests.get(f"{LM_STUDIO_BASE_URL}/v1/models", timeout=2)
    except:
        return f"⚠️ LM Studio offline", [], "LM Studio offline"
    
    models = get_models()
    if models is None:
        return "❌ Models not loaded", [], "System error"
    
    # Fast retrieval
    try:
        result = best_arch.retrieve(query, models)
        text_hits = result.get("text", [])[:3]  # Only top 3
        image_hits = result.get("image", [])[:2]  # Only top 2
    except Exception as e:
        return f"❌ Error: {str(e)}", [], "Retrieval failed"
    
    # Gallery
    gallery = []
    for h in image_hits:
        path = h.get("image_path", "")
        if path:
            full_path = IMAGES_ROOT / path
            if full_path.exists():
                gallery.append(str(full_path))
    
    # Citations
    citations = []
    seen = set()
    for h in text_hits + image_hits:
        doc = h.get("doc_name", "Unknown").replace("_", " ").title()
        page = h.get("page_id")
        key = f"{doc}_{page}"
        if key in seen:
            continue
        seen.add(key)
        page_str = f", page {page}" if page else ""
        icon = "🖼️" if h.get("image_path") else "📄"
        citations.append(f"{icon} {doc}{page_str}")
    
    citations_text = "\n".join(citations) if citations else "No sources cited"
    
    # Build SIMPLE prompt
    context = []
    
    # Add text context
    if text_hits:
        context.append("=== DOCUMENTS ===")
        for i, h in enumerate(text_hits, 1):
            doc = h.get("doc_name", "Unknown").replace("_", " ").title()
            text = h.get("text", "").strip()[:800]  # Limit text
            page = h.get("page_id", "")
            page_str = f" [Page {page}]" if page else ""
            context.append(f"\n[{i}] {doc}{page_str}:\n{text}")
    
    # Add image descriptions
    if image_hits:
        context.append("\n=== IMAGES ===")
        for i, h in enumerate(image_hits, 1):
            doc = h.get("doc_name", "Unknown").replace("_", " ").title()
            desc = h.get("text", "").strip()
            page = h.get("page_id", "")
            page_str = f" [Page {page}]" if page else ""
            context.append(f"\n[Image {i}] {doc}{page_str}:\n{desc}")
    
    context.append(f"\n=== QUESTION ===\n{query}")
    
    # Simple system prompt
    system = "You are a helpful assistant. Answer based ONLY on the documents and images above."
    
    # Generate - FAST
    try:
        answer = generate_text_only(system, "\n\n".join(context), max_tokens=512, temperature=0.1)
    except Exception as e:
        answer = f"❌ Generation error: {str(e)}"
    
    return answer, gallery, citations_text

# Simple UI
with gr.Blocks(title="Fast Document QA", theme=gr.themes.Soft()) as app:
    
    gr.Markdown("# ⚡ Fast Document Question Answering")
    gr.Markdown("Upload any PDF - get answers in seconds")
    
    with gr.Row():
        with gr.Column(scale=1):
            pdf_file = gr.File(label="Upload PDF", file_types=[".pdf"])
            pdf_name = gr.Textbox(label="Name (optional)")
            pdf_btn = gr.Button("Upload", variant="primary")
            pdf_status = gr.Textbox(label="", interactive=False)
    
    query = gr.Textbox(label="Question", placeholder="Ask about the document...", lines=2)
    ask_btn = gr.Button("Ask", variant="primary", size="lg")
    
    with gr.Row():
        with gr.Column(scale=2):
            answer = gr.Textbox(label="Answer", lines=6, interactive=False)
        with gr.Column(scale=1):
            citations = gr.Textbox(label="Sources", lines=4, interactive=False)
    
    gallery = gr.Gallery(label="Images", columns=4, rows=1, height=150)
    
    # Events
    pdf_btn.click(upload_pdf, [pdf_file, pdf_name], [pdf_status, pdf_file])
    ask_btn.click(run_pipeline, [query], [answer, gallery, citations])

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)