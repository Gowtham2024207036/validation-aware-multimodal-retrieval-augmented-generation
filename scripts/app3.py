"""
app.py  —  Clean Document QA System
Clean UI: user asks questions, gets answers with source citations.
No technical jargon shown to user.
Pipeline details hidden — only shown for demo purposes.
"""

import os, sys, json, threading, tempfile
from pathlib import Path
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
from PIL import Image

from base_retriever import SharedModels
from lmstudio_client import check_connection, generate_text_only, generate_with_images, LM_STUDIO_BASE_URL
from prompt_builder import SYSTEM_PROMPT, build_text_only_prompt, build_multimodal_prompt
import arch6_full_proposed as best_arch

IMAGES_ROOT = Path("data/raw")
_models = None
_lock   = threading.Lock()

def get_models():
    global _models
    with _lock:
        if _models is None:
            _models = SharedModels()
    return _models

def _rebuild_bm25():
    global _models
    with _lock:
        if _models is None: return
        from base_retriever import build_bm25_index
        import config
        try:
            _models.text_bm25,  _models.text_records  = build_bm25_index(_models.client, config.TEXT_COLLECTION)
            _models.image_bm25, _models.image_records = build_bm25_index(_models.client, config.IMAGE_COLLECTION)
        except Exception as e:
            print(f"BM25 rebuild: {e}")

def get_indexed_documents() -> str:
    try:
        import config
        from qdrant_client import QdrantClient
        client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
        doc_counts = defaultdict(int)
        offset = None
        while True:
            results, next_offset = client.scroll(
                collection_name=config.TEXT_COLLECTION,
                limit=200, offset=offset, with_payload=True, with_vectors=False)
            for p in results:
                dn = p.payload.get("doc_name","")
                if dn: doc_counts[dn] += 1
            if next_offset is None or not results: break
            offset = next_offset
        if not doc_counts:
            return "No documents indexed yet. Upload a PDF or image to get started."
        lines = [f"**{len(doc_counts)} document(s) in the system:**\n"]
        for doc, count in sorted(doc_counts.items()):
            readable = doc.replace("_"," ").replace("-"," ").title()
            lines.append(f"• {readable}")
        lines.append("\n_Just type your question — no need to mention the document name._")
        return "\n".join(lines)
    except Exception as e:
        return f"Could not load document list: {e}"


def delete_document(doc_name: str) -> tuple[str, str]:
    """Delete all chunks for a doc_name from Qdrant."""
    if not doc_name.strip():
        return "Please enter a document name to delete.", get_indexed_documents()
    try:
        import config
        from qdrant_client import QdrantClient
        from qdrant_client.http import models as qmodels
        client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
        for col in [config.TEXT_COLLECTION, config.IMAGE_COLLECTION]:
            client.delete(
                collection_name=col,
                points_selector=qmodels.FilterSelector(
                    filter=qmodels.Filter(must=[
                        qmodels.FieldCondition(
                            key="doc_name",
                            match=qmodels.MatchValue(value=doc_name.strip())
                        )
                    ])
                )
            )
        _rebuild_bm25()
        return f"'{doc_name}' removed from the system.", get_indexed_documents()
    except Exception as e:
        return f"Delete failed: {e}", get_indexed_documents()

def ingest_pdf_to_qdrant(file_obj, doc_name_input):
    if file_obj is None: return "No file selected.", get_indexed_documents()
    try:
        from ingest_document import ingest
        dname = doc_name_input.strip() or Path(file_obj.name).stem.replace(" ","_").upper()
        ingest(file_obj.name, dname)
        _rebuild_bm25()
        return f"'{dname}' added to the system. You can now ask questions about it.", get_indexed_documents()
    except ImportError:
        return "Install pymupdf first: pip install pymupdf", get_indexed_documents()
    except Exception as e:
        return f"Upload failed: {e}", get_indexed_documents()

def ingest_image_to_qdrant(pil_img, doc_name_input):
    if pil_img is None: return "No image selected.", get_indexed_documents()
    try:
        from ingest_document import ingest_image
        if not isinstance(pil_img, Image.Image): pil_img = Image.fromarray(pil_img)
        dname = doc_name_input.strip() or "UPLOADED_IMAGE"
        result = ingest_image(pil_img, dname, page_id=1)
        _rebuild_bm25()
        return f"Image '{dname}' added. You can now ask questions about it.", get_indexed_documents()
    except Exception as e:
        return f"Upload failed: {e}", get_indexed_documents()

def fmt_citations_clean(text_hits, image_hits):
    """Human-readable citations — just doc name and page."""
    if not text_hits and not image_hits: return ""
    seen = set()
    lines = ["**Answer found in:**"]
    for h in text_hits + image_hits:
        doc  = h.get("doc_name","?")
        page = h.get("page_id")
        key  = f"{doc}_{page}"
        if key in seen: continue
        seen.add(key)
        readable = doc.replace("_"," ").replace("-"," ").title()
        pg       = f", page {page}" if page else ""
        icon     = "📄" if h in text_hits else "🖼️"
        lines.append(f"{icon} {readable}{pg}")
    return "\n".join(lines)

def fmt_pipeline_detail(text_hits, image_hits, sub_qs):
    """Full pipeline detail for the demo accordion."""
    lines = []
    # Step 1
    lines.append("**Sub-queries generated:**")
    for i,q in enumerate(sub_qs,1): lines.append(f"{i}. {q}")
    lines.append("")
    # Step 2
    lines.append(f"**Retrieved:** {len(text_hits)} text + {len(image_hits)} image chunks")
    for h in text_hits:
        doc=h.get("doc_name","?"); page=h.get("page_id","?")
        rrf=h.get("rrf_score",0); bm25=h.get("bm25_score")
        txt=(h.get("text") or "")[:120]
        sc=f"RRF:{rrf:.3f}" + (f" BM25:{bm25:.0f}" if bm25 else "")
        lines.append(f"  📄 {doc} p.{page} [{sc}]: {txt}...")
    for h in image_hits:
        doc=h.get("doc_name","?"); page=h.get("page_id","?")
        conf=h.get("cde_confidence"); rrf=h.get("rrf_score",0)
        lines.append(f"  🖼️ {doc} p.{page} [RRF:{rrf:.3f}" + (f" CDE:{conf:.2f}" if conf else "") + "]")
    # Step 3
    all_hits = [(h,"text") for h in text_hits]+[(h,"img") for h in image_hits]
    if any(h.get("cde_confidence") is not None for h,_ in all_hits):
        lines.append("")
        lines.append("**CDE Confidence Scores (DistilBERT F1=0.9274):**")
        for h,mod in all_hits:
            conf=h.get("cde_confidence","—"); rel=h.get("cde_relevance","—")
            doc=h.get("doc_name","?")
            icon="📄" if mod=="text" else "🖼️"
            cs=f"{conf:.3f}" if isinstance(conf,float) else str(conf)
            rs=f"{rel:.3f}"  if isinstance(rel, float) else str(rel)
            lines.append(f"  {icon} {doc}: relevance={rs} confidence={cs}")
    return "\n".join(lines)

def run_pipeline(query, mode):
    if not query.strip():
        return "Please type a question.", [], "", "", ""
    models = get_models()
    try:
        result     = best_arch.retrieve(query, models)
        text_hits  = result.get("text",  [])
        image_hits = result.get("image", [])
        sub_qs     = result.get("sub_queries", [query])
    except Exception as e:
        return f"Error: {e}", [], "", "", "ERROR"

    gallery = []
    for h in image_hits:
        path = h.get("image_path","")
        doc  = h.get("doc_name","").replace("_"," ").title()
        page = h.get("page_id")
        cap  = f"{doc}" + (f" p.{page}" if page else "")
        full = IMAGES_ROOT / path if path else None
        gallery.append((str(full) if full and full.exists() else None, cap))

    if not check_connection():
        ans = f"LM Studio is offline. Please start it at {LM_STUDIO_BASE_URL}."
        return ans, gallery, fmt_citations_clean(text_hits, image_hits), \
               fmt_pipeline_detail(text_hits, image_hits, sub_qs), "LM Studio OFFLINE"

    high_conf = [h for h in image_hits if h.get("cde_confidence",0)>0.5 or h.get("rrf_score",0)>0.01]
    use_multi = (mode=="Multimodal") or bool(high_conf)

    try:
        if use_multi and image_hits:
            ut, ip = build_multimodal_prompt(query, text_hits, image_hits)
            ans    = generate_with_images(SYSTEM_PROMPT, ut, ip, str(IMAGES_ROOT), 1024, 0.1)
        else:
            ut, _ = build_text_only_prompt(query, text_hits, image_hits)
            ans   = generate_text_only(SYSTEM_PROMPT, ut, 1024, 0.1)
    except Exception as e:
        ans = f"Generation error: {e}"

    status = f"{len(text_hits)} text + {len(image_hits)} images retrieved"
    return ans, gallery, fmt_citations_clean(text_hits, image_hits), \
           fmt_pipeline_detail(text_hits, image_hits, sub_qs), status

SAMPLES = [
    "What scholarship is available for SC ST students?",
    "What is the minimum percentage required to apply for TNEA?",
    "What is the long-term debt to total liabilities for COSTCO in FY2021?",
    "What are the total assets for COSTCO as of August 2021?",
]

with gr.Blocks(title="Document QA", theme=gr.themes.Soft(),
               css=".answer textarea{font-size:15px!important;line-height:1.8!important}") as app:

    gr.Markdown("# 📚 Document Question Answering\nAsk questions about any uploaded document. The system finds the answer and tells you its source.")

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### Available Documents")
            docs_display = gr.Markdown(value=get_indexed_documents())
            gr.Button("↻ Refresh", size="sm", variant="secondary").click(
                fn=get_indexed_documents, outputs=docs_display)

        with gr.Column(scale=1):
            gr.Markdown("### Add Document")
            with gr.Tab("PDF"):
                pdf_f   = gr.File(label="Select PDF", file_types=[".pdf"])
                pdf_n   = gr.Textbox(label="Name (optional)", placeholder="e.g. TNEA_2025")
                pdf_btn = gr.Button("Upload", variant="primary")
                pdf_s   = gr.Textbox(label="", interactive=False, lines=2)
                pdf_btn.click(fn=ingest_pdf_to_qdrant, inputs=[pdf_f, pdf_n], outputs=[pdf_s, docs_display])
            with gr.Tab("Image"):
                img_f   = gr.Image(label="Select image", type="pil", height=150, sources=["upload","clipboard"])
                img_n   = gr.Textbox(label="Name", placeholder="e.g. Chart_2024")
                img_btn = gr.Button("Upload", variant="primary")
                img_s   = gr.Textbox(label="", interactive=False, lines=2)
                img_btn.click(fn=ingest_image_to_qdrant, inputs=[img_f, img_n], outputs=[img_s, docs_display])
            with gr.Tab("Remove Document"):
                gr.Markdown("Enter the exact document name (as shown in Available Documents) to remove it.")
                del_name = gr.Textbox(label="Document name to remove",
                                      placeholder="e.g. 2_INFORMATION_BROCHURE")
                del_btn  = gr.Button("Remove Document", variant="stop")
                del_s    = gr.Textbox(label="", interactive=False, lines=2)
                del_btn.click(fn=delete_document, inputs=[del_name],
                              outputs=[del_s, docs_display])

    gr.Markdown("---")

    with gr.Row():
        with gr.Column(scale=3):
            qbox = gr.Textbox(label="Your question", lines=2,
                              placeholder="e.g. What scholarships are available for SC ST students?")
            gr.Examples(examples=SAMPLES, inputs=qbox)
        with gr.Column(scale=1):
            mode = gr.Radio(["Multimodal","Text only"], value="Multimodal", label="Mode")
            run_btn    = gr.Button("Get Answer", variant="primary", size="lg")
            status_box = gr.Textbox(label="", interactive=False, lines=1)

    # Answer — primary output, clean
    ans_box = gr.Textbox(label="Answer", lines=7, interactive=False,
                         elem_classes=["answer"],
                         placeholder="Answer appears here...")
    cit_out = gr.Markdown()

    # Retrieved images
    with gr.Row():
        img_gal = gr.Gallery(label="Retrieved images", columns=4, height=200,
                             object_fit="contain", show_label=True)

    # Pipeline details — hidden by default, for demo only
    with gr.Accordion("🔬 Pipeline Details (for demo)", open=False):
        pipeline_md = gr.Markdown("_Run a query to see pipeline details._")

    run_btn.click(
        fn=run_pipeline,
        inputs=[qbox, mode],
        outputs=[ans_box, img_gal, cit_out, pipeline_md, status_box],
    )

if __name__=="__main__":
    print("Loading models..."); get_models()
    print(f"LM Studio: {'ONLINE' if check_connection() else 'OFFLINE'}")
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)