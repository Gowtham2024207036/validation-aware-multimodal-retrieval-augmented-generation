"""
app.py  —  Clean Document QA System with Beautiful UI
Compatible with Gradio 6.0+
"""

import os
import sys
import threading
from pathlib import Path
from collections import defaultdict
import base64
from io import BytesIO
import time
import traceback

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
from PIL import Image
import markdown

from base_retriever import SharedModels
from lmstudio_client import check_connection, generate_text_only, generate_with_images, LM_STUDIO_BASE_URL
from prompt_builder import SYSTEM_PROMPT, build_text_only_prompt, build_multimodal_prompt
import arch6_full_proposed as best_arch

# Import ingest functions
try:
    from ingest_document import ingest_pdf, ingest_image
except ImportError:
    # Try alternative import
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from ingest_document import ingest_pdf, ingest_image

import config

IMAGES_ROOT = Path(config.RAW_DATA_DIR)
_models = None
_lock = threading.Lock()

# Custom CSS for beautiful UI
CUSTOM_CSS = """
:root {
    --primary-color: #4361ee;
    --primary-light: #4895ef;
    --secondary-color: #3f37c9;
    --success-color: #4cc9f0;
    --warning-color: #f72585;
    --dark-bg: #1e1e2e;
    --light-bg: #f8f9fa;
    --text-dark: #2b2d42;
    --text-light: #8d99ae;
    --border-radius: 12px;
    --box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
}

.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
    padding: 20px;
}

.header {
    text-align: center;
    padding: 2rem 0;
    background: white;
    border-radius: var(--border-radius);
    box-shadow: var(--box-shadow);
    margin-bottom: 2rem;
    animation: fadeIn 0.5s ease;
}

.header h1 {
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--primary-color), var(--warning-color));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}

.header p {
    color: var(--text-light);
    font-size: 1.1rem;
}

.upload-section {
    background: white;
    border-radius: var(--border-radius);
    padding: 1.5rem;
    box-shadow: var(--box-shadow);
    margin-bottom: 2rem;
    animation: fadeIn 0.6s ease;
}

.upload-tabs {
    border: none !important;
}

.upload-tabs .tab-nav {
    background: var(--light-bg);
    border-radius: 8px;
    padding: 4px;
    margin-bottom: 10px;
}

.upload-tabs .tab-nav button {
    border-radius: 6px !important;
    border: none !important;
    background: transparent !important;
    color: var(--text-dark) !important;
    font-weight: 500 !important;
    transition: all 0.3s ease;
}

.upload-tabs .tab-nav button.selected {
    background: var(--primary-color) !important;
    color: white !important;
}

.upload-area {
    border: 2px dashed var(--primary-light);
    border-radius: var(--border-radius);
    padding: 1.5rem;
    text-align: center;
    transition: all 0.3s ease;
    background: var(--light-bg);
    margin-bottom: 10px;
}

.upload-area:hover {
    border-color: var(--primary-color);
    background: white;
}

.upload-btn {
    background: linear-gradient(135deg, var(--primary-color), var(--secondary-color)) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    font-weight: 600 !important;
    transition: transform 0.3s ease !important;
    width: 100%;
}

.upload-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(67, 97, 238, 0.3);
}

.query-section {
    background: white;
    border-radius: var(--border-radius);
    padding: 2rem;
    box-shadow: var(--box-shadow);
    margin-bottom: 2rem;
    animation: fadeIn 0.7s ease;
}

.query-box textarea {
    border: 2px solid var(--light-bg) !important;
    border-radius: 12px !important;
    padding: 15px !important;
    font-size: 1rem !important;
    transition: all 0.3s ease !important;
}

.query-box textarea:focus {
    border-color: var(--primary-color) !important;
    box-shadow: 0 0 0 3px rgba(67, 97, 238, 0.1) !important;
}

.ask-btn {
    background: linear-gradient(135deg, var(--success-color), var(--primary-color)) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 30px !important;
    font-weight: 600 !important;
    font-size: 1.1rem !important;
    transition: all 0.3s ease !important;
    width: 100%;
}

.ask-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 20px rgba(76, 201, 240, 0.4);
}

.answer-section {
    background: white;
    border-radius: var(--border-radius);
    padding: 2rem;
    box-shadow: var(--box-shadow);
    margin-bottom: 2rem;
    animation: fadeIn 0.8s ease;
}

.answer-box {
    background: var(--light-bg);
    border-radius: 12px;
    padding: 1.5rem;
    border-left: 4px solid var(--primary-color);
    min-height: 150px;
}

.answer-box p {
    margin: 0.5rem 0;
    line-height: 1.8;
}

.answer-box pre {
    background: #2d2d2d;
    color: #f8f8f2;
    padding: 1rem;
    border-radius: 8px;
    overflow-x: auto;
}

.answer-box code {
    font-family: 'Courier New', monospace;
}

.citations {
    background: linear-gradient(135deg, #f6f9fc, #f1f4f8);
    border-radius: 10px;
    padding: 1rem;
    margin-top: 1rem;
    border-left: 4px solid var(--success-color);
}

.citations p {
    margin: 0.3rem 0;
    color: var(--text-dark);
    font-size: 0.95rem;
}

.gallery-section {
    background: white;
    border-radius: var(--border-radius);
    padding: 2rem;
    box-shadow: var(--box-shadow);
    margin-bottom: 2rem;
    animation: fadeIn 0.9s ease;
}

.gallery-section .gallery {
    min-height: 200px;
}

.detail-accordion {
    margin-top: 1rem;
    border: 1px solid var(--light-bg) !important;
    border-radius: 10px !important;
    overflow: hidden;
    animation: fadeIn 1s ease;
}

.detail-accordion .label-wrap {
    background: var(--light-bg) !important;
    padding: 1rem !important;
    font-weight: 600 !important;
    cursor: pointer;
}

.detail-accordion .label-wrap span {
    color: var(--primary-color) !important;
}

.detail-content {
    background: white;
    padding: 1rem;
    font-family: monospace;
    font-size: 0.9rem;
    line-height: 1.6;
    max-height: 300px;
    overflow-y: auto;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.fade-in {
    animation: fadeIn 0.5s ease forwards;
}

.gr-button {
    margin-top: 0.5rem !important;
}

.gr-form {
    background: transparent !important;
}

footer {
    display: none !important;
}

.status-success {
    background: #d4edda;
    color: #155724;
    border: 1px solid #c3e6cb;
    padding: 10px;
    border-radius: 8px;
    margin-top: 10px;
}

.status-error {
    background: #f8d7da;
    color: #721c24;
    border: 1px solid #f5c6cb;
    padding: 10px;
    border-radius: 8px;
    margin-top: 10px;
}

.status-warning {
    background: #fff3cd;
    color: #856404;
    border: 1px solid #ffeeba;
    padding: 10px;
    border-radius: 8px;
    margin-top: 10px;
}
"""


def get_models():
    """Get or initialize shared models"""
    global _models
    with _lock:
        if _models is None:
            try:
                _models = SharedModels()
                print("✅ Models loaded successfully")
            except Exception as e:
                print(f"❌ Error loading models: {e}")
                traceback.print_exc()
                return None
    return _models


def rebuild_bm25():
    """Rebuild BM25 indexes after document changes"""
    global _models
    with _lock:
        if _models is None:
            return
        from base_retriever import build_bm25_index
        try:
            _models.text_bm25, _models.text_records = build_bm25_index(
                _models.client, config.TEXT_COLLECTION
            )
            _models.image_bm25, _models.image_records = build_bm25_index(
                _models.client, config.IMAGE_COLLECTION
            )
            print("✅ BM25 indexes rebuilt")
        except Exception as e:
            print(f"⚠️ BM25 rebuild error: {e}")


def upload_pdf(file_obj, doc_name_input):
    """Handle PDF upload"""
    if file_obj is None:
        return (
            gr.Textbox(value="❌ No file selected", visible=True),
            gr.File(value=None)
        )
    
    doc_name = doc_name_input.strip() or Path(file_obj.name).stem.replace(" ", "_").upper()
    
    # Process upload
    try:
        success, message = ingest_pdf(file_obj.name, doc_name)
        
        if success:
            rebuild_bm25()
            return (
                gr.Textbox(value=f"✅ {message}", visible=True),
                gr.File(value=None)
            )
        else:
            return (
                gr.Textbox(value=f"❌ {message}", visible=True),
                gr.File(value=None)
            )
    except Exception as e:
        traceback.print_exc()
        return (
            gr.Textbox(value=f"❌ Error: {str(e)}", visible=True),
            gr.File(value=None)
        )


def upload_image(pil_img, doc_name_input):
    """Handle image upload"""
    if pil_img is None:
        return (
            gr.Textbox(value="❌ No image selected", visible=True),
            gr.Image(value=None)
        )
    
    doc_name = doc_name_input.strip() or "UPLOADED_IMAGE"
    
    try:
        if not isinstance(pil_img, Image.Image):
            pil_img = Image.fromarray(pil_img)
        
        result = ingest_image(pil_img, doc_name, page_id=1)
        rebuild_bm25()
        
        if result.get("success"):
            return (
                gr.Textbox(value=f"✅ {result['message']}", visible=True),
                gr.Image(value=None)
            )
        else:
            return (
                gr.Textbox(value=f"❌ {result['message']}", visible=True),
                gr.Image(value=None)
            )
    except Exception as e:
        traceback.print_exc()
        return (
            gr.Textbox(value=f"❌ Upload failed: {str(e)}", visible=True),
            gr.Image(value=None)
        )


def format_answer_with_markdown(answer):
    """Convert answer to HTML with markdown"""
    if not answer:
        return "<p><i>No answer generated yet.</i></p>"
    
    # Convert markdown to HTML
    html = markdown.markdown(
        answer,
        extensions=['extra', 'codehilite', 'tables']
    )
    
    # Wrap in div with styling
    return f'<div class="answer-content">{html}</div>'


def create_image_gallery(image_hits):
    """Create gallery items from image hits"""
    gallery_items = []
    
    for h in image_hits:
        path = h.get("image_path", "")
        doc = h.get("doc_name", "").replace("_", " ").title()
        page = h.get("page_id")
        
        if not path:
            continue
            
        full_path = IMAGES_ROOT / path
        if not full_path.exists():
            # Try alternative path
            alt_path = Path("data/raw") / path
            if alt_path.exists():
                full_path = alt_path
            else:
                continue
        
        # Create caption
        caption = f"{doc}"
        if page:
            caption += f" (Page {page})"
        
        # Add confidence if available
        if h.get("cde_confidence"):
            confidence = h['cde_confidence']
            caption += f" • {confidence:.1%} confidence"
        
        gallery_items.append((str(full_path), caption))
    
    return gallery_items


def run_pipeline(query, mode, history):
    """Run the full RAG pipeline"""
    print(f"🔍 Processing query: {query}")
    
    if not query.strip():
        return (
            gr.HTML(value="<p><i>Please enter a question.</i></p>"),
            [],
            gr.HTML(value="<p><i>No sources cited</i></p>"),
            gr.HTML(value="<i>No pipeline details yet.</i>"),
            gr.Textbox(value="⚠️ No query provided", visible=True),
            history if history else []
        )
    
    # Check LM Studio
    lm_online = check_connection()
    if not lm_online:
        return (
            gr.HTML(value=f"<p>⚠️ LM Studio is offline. Please start it at {LM_STUDIO_BASE_URL}</p>"),
            [],
            gr.HTML(value="<p><i>LM Studio offline</i></p>"),
            gr.HTML(value="<i>LM Studio offline</i>"),
            gr.Textbox(value="🔴 LM Studio Offline", visible=True),
            history if history else []
        )
    
    models = get_models()
    if models is None:
        return (
            gr.HTML(value="<p>❌ Failed to load retrieval models. Check Qdrant connection.</p>"),
            [],
            gr.HTML(value="<p><i>System error</i></p>"),
            gr.HTML(value="<i>Model loading failed</i>"),
            gr.Textbox(value="🔴 System Error", visible=True),
            history if history else []
        )
    
    try:
        result = best_arch.retrieve(query, models)
        text_hits = result.get("text", [])
        image_hits = result.get("image", [])
        sub_qs = result.get("sub_queries", [query])
        print(f"📚 Retrieved: {len(text_hits)} text, {len(image_hits)} images")
    except Exception as e:
        traceback.print_exc()
        return (
            gr.HTML(value=f"<p>❌ Error in retrieval: {str(e)}</p>"),
            [],
            gr.HTML(value="<p><i>Retrieval failed</i></p>"),
            gr.HTML(value=f"<i>Error: {str(e)}</i>"),
            gr.Textbox(value="🔴 Retrieval Error", visible=True),
            history if history else []
        )
    
    # Create gallery
    gallery = create_image_gallery(image_hits)
    
    # Format citations
    citations = []
    seen = set()
    for h in text_hits + image_hits:
        doc = h.get("doc_name", "Unknown")
        page = h.get("page_id")
        key = f"{doc}_{page}"
        if key in seen:
            continue
        seen.add(key)
        
        readable = doc.replace("_", " ").replace("-", " ").title()
        page_str = f", page {page}" if page else ""
        
        # Determine icon
        if h.get("image_path"):
            icon = "🖼️"
        else:
            icon = "📄"
        
        citations.append(f"{icon} <strong>{readable}</strong>{page_str}")
    
    citations_html = "<br>".join(citations) if citations else "<p><i>No sources cited</i></p>"
    
    # Generate answer
    try:
        # Decide whether to use multimodal
        has_confident_images = any(
            h.get("cde_confidence", 0) > 0.3 for h in image_hits
        )
        use_multi = mode == "Multimodal" and image_hits and has_confident_images
        
        if use_multi and image_hits:
            user_text, img_paths = build_multimodal_prompt(query, text_hits, image_hits)
            answer = generate_with_images(
                SYSTEM_PROMPT, user_text, img_paths, str(IMAGES_ROOT), 1024, 0.1
            )
        else:
            user_text, _ = build_text_only_prompt(query, text_hits, image_hits)
            answer = generate_text_only(SYSTEM_PROMPT, user_text, 1024, 0.1)
        
        print(f"✅ Answer generated: {answer[:100]}...")
        
        # Format answer with markdown
        answer_html = format_answer_with_markdown(answer)
        
        # Initialize history if None
        if history is None:
            history = []
        
        # For Gradio 6.0, history should be a list of tuples (user, assistant)
        # or a list of lists [user_message, assistant_message]
        new_history = list(history) if history else []
        new_history.append((query, answer))  # Use tuple format
        
    except Exception as e:
        traceback.print_exc()
        answer_html = f"<p>❌ Generation error: {str(e)}</p>"
        new_history = history if history else []
    
    # Pipeline details
    detail_lines = []
    detail_lines.append("### 🔍 Sub-queries Generated")
    for i, q in enumerate(sub_qs, 1):
        detail_lines.append(f"{i}. {q}")
    
    detail_lines.append("\n### 📚 Retrieved Chunks")
    detail_lines.append(f"- **Text chunks:** {len(text_hits)}")
    detail_lines.append(f"- **Image chunks:** {len(image_hits)}")
    
    if text_hits:
        detail_lines.append("\n**Top text chunks:**")
        for h in text_hits[:3]:
            doc = h.get("doc_name", "?")
            page = h.get("page_id", "?")
            text = (h.get("text") or "")[:100]
            conf = h.get("cde_confidence", 0)
            detail_lines.append(f"  • {doc} p.{page} [conf:{conf:.2f}]: {text}...")
    
    if image_hits:
        detail_lines.append("\n**Top images:**")
        for h in image_hits[:3]:
            doc = h.get("doc_name", "?")
            page = h.get("page_id", "?")
            conf = h.get("cde_confidence", 0)
            detail_lines.append(f"  • {doc} p.{page} [conf:{conf:.2f}]")
    
    detail_html = "<br>".join(detail_lines)
    
    status_text = f"✅ Retrieved {len(text_hits)} text + {len(image_hits)} images"
    
    return (
        gr.HTML(value=answer_html),
        gallery,
        gr.HTML(value=citations_html),
        gr.HTML(value=detail_html),
        gr.Textbox(value=status_text, visible=True),
        new_history  # List of tuples: [(user1, assistant1), (user2, assistant2)]
    )


def clear_upload_status():
    """Clear upload status messages"""
    return gr.Textbox(value="", visible=False)


# Example questions
SAMPLE_QUESTIONS = [
    "What scholarship is available for SC ST students?",
    "What is the minimum percentage required for TNEA?",
    "What is COSTCO's long-term debt in FY2021?",
    "What are COSTCO's total assets as of August 2021?",
    "Show me the revenue chart for 2021",
    "Compare net income between 2020 and 2021",
]


# Create the Gradio app - remove theme from here
with gr.Blocks(title="📚 Document QA System") as app:
    
    # Header
    with gr.Column(elem_classes="header"):
        gr.HTML("""
            <h1>📚 Document Question Answering</h1>
            <p>Ask questions about your documents — get answers with sources</p>
        """)
    
    # Upload Section
    with gr.Column(elem_classes="upload-section"):
        gr.Markdown("### 📤 Add Documents")
        
        with gr.Tabs(elem_classes="upload-tabs"):
            with gr.TabItem("📄 PDF"):
                with gr.Column():
                    pdf_file = gr.File(
                        label="Select PDF",
                        file_types=[".pdf"],
                        elem_classes="upload-area"
                    )
                    pdf_name = gr.Textbox(
                        label="Document Name (optional)",
                        placeholder="e.g. TNEA_2025",
                        lines=1
                    )
                    pdf_btn = gr.Button(
                        "Upload PDF",
                        variant="primary",
                        elem_classes="upload-btn"
                    )
                    pdf_status = gr.Textbox(
                        label="",
                        visible=False,
                        lines=2
                    )
            
            with gr.TabItem("🖼️ Image"):
                with gr.Column():
                    img_file = gr.Image(
                        label="Select Image",
                        type="pil",
                        height=200,
                        sources=["upload", "clipboard"],
                        elem_classes="upload-area"
                    )
                    img_name = gr.Textbox(
                        label="Document Name",
                        placeholder="e.g. Chart_2024",
                        lines=1
                    )
                    img_btn = gr.Button(
                        "Upload Image",
                        variant="primary",
                        elem_classes="upload-btn"
                    )
                    img_status = gr.Textbox(
                        label="",
                        visible=False,
                        lines=2
                    )
    
    # Query Section
    with gr.Column(elem_classes="query-section"):
        gr.Markdown("### ❓ Ask a Question")
        
        # Use standard Chatbot without type parameter
        chatbot = gr.Chatbot(label="Conversation History", height=300)
        
        with gr.Row():
            with gr.Column(scale=4):
                query = gr.Textbox(
                    label="Your Question",
                    placeholder="e.g. What scholarships are available for SC ST students?",
                    lines=2,
                    elem_classes="query-box"
                )
            
            with gr.Column(scale=1, min_width=150):
                mode = gr.Radio(
                    ["Multimodal", "Text only"],
                    value="Multimodal",
                    label="Mode"
                )
                ask_btn = gr.Button(
                    "Ask Question",
                    variant="primary",
                    elem_classes="ask-btn"
                )
        
        # Examples
        gr.Examples(
            examples=SAMPLE_QUESTIONS,
            inputs=query,
            outputs=None,
            label="Try these examples"
        )
    
    # Answer Section
    with gr.Column(elem_classes="answer-section"):
        gr.Markdown("### 💡 Current Answer")
        
        answer = gr.HTML(
            label="",
            elem_classes="answer-box"
        )
        
        citations = gr.HTML(
            label="Sources",
            elem_classes="citations"
        )
    
    # Gallery Section
    with gr.Column(elem_classes="gallery-section"):
        gr.Markdown("### 🖼️ Retrieved Images")
        
        gallery = gr.Gallery(
            label="",
            columns=4,
            rows=1,
            height=200,
            object_fit="contain"
        )
    
    # Status and Details
    status = gr.Textbox(
        label="",
        visible=False
    )
    
    with gr.Accordion("🔬 Pipeline Details", open=False, elem_classes="detail-accordion"):
        pipeline_detail = gr.HTML(
            value="<i>Run a query to see pipeline details.</i>",
            elem_classes="detail-content"
        )
    
    # Event handlers
    pdf_btn.click(
        fn=upload_pdf,
        inputs=[pdf_file, pdf_name],
        outputs=[pdf_status, pdf_file]
    )
    
    img_btn.click(
        fn=upload_image,
        inputs=[img_file, img_name],
        outputs=[img_status, img_file]
    )
    
    # Chain the events for ask button
    ask_event = ask_btn.click(
        fn=run_pipeline,
        inputs=[query, mode, chatbot],
        outputs=[answer, gallery, citations, pipeline_detail, status, chatbot]
    )
    
    ask_event.then(
        fn=lambda: "",  # Clear query after asking
        inputs=None,
        outputs=[query]
    )


if __name__ == "__main__":
    print("=" * 50)
    print("🚀 Starting Document QA System")
    print("=" * 50)
    
    print("\n📡 Checking services...")
    
    # Check Qdrant
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT, timeout=2)
        client.get_collections()
        print("✅ Qdrant: Connected")
    except Exception as e:
        print(f"⚠️ Qdrant: Not connected - {e}")
        print("   Run: docker-compose up -d")
    
    # Check LM Studio
    lm_status = "🟢 ONLINE" if check_connection() else "🔴 OFFLINE"
    print(f"📡 LM Studio: {lm_status}")
    
    # Load models
    print("\n🔄 Loading models...")
    try:
        get_models()
    except Exception as e:
        print(f"⚠️ Warning: Could not load models: {e}")
        print("   The app will still run, but retrieval may not work.")
    
    print(f"\n🌐 Starting server at http://localhost:7860")
    print("   Press Ctrl+C to stop")
    print("=" * 50)
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="purple",
            neutral_hue="gray",
        ),
        css=CUSTOM_CSS
    )