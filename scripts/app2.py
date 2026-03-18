"""
app.py  —  Multimodal RAG Full Pipeline Demo
---------------------------------------------
Works for ANY document type: financial reports, research papers,
school records, charts, tables, academic papers, etc.

CORRECT FLOW (matches architecture diagram):
  1. [Optional] Upload image/PDF → embed → store in Qdrant
  2. Enter query → Query Expansion → Hybrid Retrieval (BM25+Vector+RRF)
                → CDE Validation → Qwen2.5-VL → Answer with citations

Run:
    pip install gradio pymupdf
    python scripts/app.py  →  http://localhost:7860
"""


import os, sys, json, threading, tempfile
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
from PIL import Image

from base_retriever import SharedModels
from lmstudio_client import (check_connection, generate_text_only,
                              generate_with_images, LM_STUDIO_BASE_URL)
from prompt_builder import SYSTEM_PROMPT, build_text_only_prompt, build_multimodal_prompt
import arch6_full_proposed as best_arch

IMAGES_ROOT = Path("data/raw")
_models     = None
_lock       = threading.Lock()

def get_models():
    global _models
    with _lock:
        if _models is None:
            _models = SharedModels()
    return _models


# ------------------------------------------------------------------ #
# Ingestion — image uploaded by user → embed → store in Qdrant
# ------------------------------------------------------------------ #

def ingest_image_to_qdrant(pil_img, doc_name: str) -> str:
    """
    Embed uploaded image with CLIP and store in Qdrant.
    Also creates a text chunk so BM25 can find it by doc_name.
    """
    if pil_img is None:
        return "No image uploaded."
    try:
        from ingest_document import ingest_image
        if not isinstance(pil_img, Image.Image):
            pil_img = Image.fromarray(pil_img)
        dname  = doc_name.strip() or "UPLOADED_IMAGE"
        result = ingest_image(pil_img, dname, page_id=1)
        return (
            f"Image embedded and stored in Qdrant.\n"
            f"  doc_name : {result['doc_name']}\n"
            f"  image_id : {result['image_id']}\n"
            f"  text_id  : {result['text_id']}\n"
            f"  path     : {result['rel_path']}\n\n"
            f"Now query it: describe what is shown in {dname}"
        )
    except Exception as e:
        return f"Ingestion failed: {e}"


def ingest_pdf_to_qdrant(file_obj, doc_name_input: str) -> str:
    """Parse PDF pages → chunk text → embed → store all in Qdrant."""
    if file_obj is None:
        return "No file uploaded."
    try:
        from ingest_document import ingest
        dname = doc_name_input.strip() or \
                Path(file_obj.name).stem.replace(" ", "_").upper()
        ingest(file_obj.name, dname)
        return (
            f"PDF indexed into Qdrant successfully.\n"
            f"  doc_name : {dname}\n"
            f"Text chunks and page images are now searchable.\n\n"
            f"Query it: ask any question about {dname}"
        )
    except ImportError:
        return "pymupdf not installed. Run: pip install pymupdf"
    except Exception as e:
        return f"PDF ingestion failed: {e}"


# ------------------------------------------------------------------ #
# Intermediate output formatters
# ------------------------------------------------------------------ #

def fmt_step1(sub_queries: list) -> str:
    if not sub_queries:
        return "_No sub-queries generated._"
    lines = ["**Step 1 — Query Expansion**\n"]
    for i, q in enumerate(sub_queries, 1):
        lines.append(f"{i}. {q}")
    return "\n".join(lines)


def fmt_step2(text_hits: list, image_hits: list) -> str:
    lines = [f"**Step 2 — Hybrid Retrieval (BM25 + Vector + RRF)**\n"
             f"Retrieved: {len(text_hits)} text + {len(image_hits)} image chunks\n"]

    for i, h in enumerate(text_hits, 1):
        doc  = h.get("doc_name", "?")
        page = h.get("page_id")
        qid  = h.get("quote_id", "")
        rrf  = h.get("rrf_score", h.get("score", 0))
        bm25 = h.get("bm25_score")
        vec  = h.get("vector_score")
        txt  = (h.get("text") or "")[:200]
        pg   = f"Page {page}" if page else qid
        sc   = [f"RRF:{rrf:.4f}"]
        if bm25: sc.append(f"BM25:{bm25:.1f}")
        if vec:  sc.append(f"Vec:{vec:.4f}")
        lines.append(f"\n**[Text {i}]** `{doc}` — {pg}")
        lines.append(f"_{' | '.join(sc)}_\n> {txt}...")

    for i, h in enumerate(image_hits, 1):
        doc  = h.get("doc_name", "?")
        page = h.get("page_id")
        path = h.get("image_path", "")
        rrf  = h.get("rrf_score", 0)
        pg   = f"Page {page}" if page else path.split("/")[-1]
        lines.append(f"\n**[Image {i}]** `{doc}` — {pg}")
        lines.append(f"_RRF:{rrf:.4f}_")

    return "\n".join(lines)


def fmt_step3(text_hits: list, image_hits: list) -> str:
    all_hits = [(h, "text") for h in text_hits] + \
               [(h, "img")  for h in image_hits]
    has_cde  = any(h.get("cde_confidence") is not None for h, _ in all_hits)

    if not has_cde:
        return ("**Step 3 — CDE Validation**\n"
                "_CDE scores available when using Architecture 6._")

    lines = ["**Step 3 — CDE Validation** (DistilBERT F1=0.9274)\n"]
    lines.append("| # | Type | Document | Page | Relevance | Confidence | Status |")
    lines.append("|---|------|----------|------|-----------|------------|--------|")

    for i, (h, mod) in enumerate(all_hits, 1):
        doc  = h.get("doc_name", "?")[:25]
        page = h.get("page_id")
        pg   = str(page) if page else "—"
        rel  = h.get("cde_relevance",  "—")
        conf = h.get("cde_confidence", "—")
        red  = "❌ removed" if h.get("cde_redundant") else "✅ kept"
        icon = "📄" if mod == "text" else "🖼️"
        rs   = f"{rel:.3f}"       if isinstance(rel,  float) else str(rel)
        cs   = f"**{conf:.3f}**"  if isinstance(conf, float) else str(conf)
        lines.append(f"| {i} | {icon} | {doc} | {pg} | {rs} | {cs} | {red} |")

    return "\n".join(lines)


def fmt_citations(text_hits: list, image_hits: list) -> str:
    if not text_hits and not image_hits:
        return ""
    lines = ["**Citations:**"]
    for i, h in enumerate(text_hits, 1):
        doc  = h.get("doc_name", "?")
        page = h.get("page_id")
        qid  = h.get("quote_id", "")
        pg   = f"Page {page}" if page else qid
        lines.append(f"📄 [{i}] `{doc}` — {pg}")
    for i, h in enumerate(image_hits, 1):
        doc  = h.get("doc_name", "?")
        page = h.get("page_id")
        path = h.get("image_path", "").split("/")[-1]
        pg   = f"Page {page}" if page else path
        lines.append(f"🖼️ [Img {i}] `{doc}` — {pg}")
    return "\n".join(lines)


# ------------------------------------------------------------------ #
# Main pipeline
# ------------------------------------------------------------------ #

def run_pipeline(query: str, mode: str):
    """Full RAG pipeline — retrieves from Qdrant for any query."""
    if not query.strip():
        e = "_Run a query to see results._"
        return e, e, e, [], "Please enter a question.", "", ""

    models = get_models()

    try:
        result     = best_arch.retrieve(query, models)
        text_hits  = result.get("text",  [])
        image_hits = result.get("image", [])
        sub_qs     = result.get("sub_queries", [query])
    except Exception as e:
        err = f"Retrieval error: {e}"
        return err, err, err, [], err, "", "ERROR"

    s1 = fmt_step1(sub_qs)
    s2 = fmt_step2(text_hits, image_hits)
    s3 = fmt_step3(text_hits, image_hits)

    # Build gallery — actual image files from Qdrant
    gallery = []
    for h in image_hits:
        path = h.get("image_path", "")
        doc  = h.get("doc_name", "")
        page = h.get("page_id")
        conf = h.get("cde_confidence")
        pg   = f"p.{page}" if page else path.split("/")[-1]
        cap  = f"{doc} {pg}" + (f" | CDE:{conf:.2f}" if conf else "")
        full = IMAGES_ROOT / path if path else None
        gallery.append((str(full) if full and full.exists() else None, cap))

    if not check_connection():
        ans = f"LM Studio offline — start it at {LM_STUDIO_BASE_URL}"
        return s1, s2, s3, gallery, ans, fmt_citations(text_hits, image_hits), \
               "Retrieval OK | LM Studio OFFLINE"

    # Auto-upgrade to multimodal when high-confidence images retrieved
    high_conf_imgs = [h for h in image_hits if h.get("cde_confidence", 0) > 0.5
                      or h.get("rrf_score", 0) > 0.01]
    use_multimodal = (mode == "Multimodal") or bool(high_conf_imgs)

    try:
        if use_multimodal and image_hits:
            ut, ip = build_multimodal_prompt(query, text_hits, image_hits)
            ans    = generate_with_images(SYSTEM_PROMPT, ut, ip,
                                          str(IMAGES_ROOT), 1024, 0.1)
            if mode == "Text only" and high_conf_imgs:
                ans = "[Auto-upgraded to Multimodal — images found]\n\n" + ans
        else:
            ut, _ = build_text_only_prompt(query, text_hits, image_hits)
            ans   = generate_text_only(SYSTEM_PROMPT, ut, 1024, 0.1)
    except Exception as e:
        ans = f"Generation error: {e}"

    status = (f"Arch 6 (Full CDE) | {len(sub_qs)} sub-queries | "
              f"{len(text_hits)} text + {len(image_hits)} images | {mode}")
    return s1, s2, s3, gallery, ans, fmt_citations(text_hits, image_hits), status


# ------------------------------------------------------------------ #
# Gradio UI
# ------------------------------------------------------------------ #

SAMPLES = [
    "What is the long-term debt to total liabilities for COSTCO in FY2021?",
    "What are the total assets for COSTCO as of August 2021?",
    "What is the net income for COSTCO in FY2021?",
    "How do SWEM models compare to CNN in terms of accuracy?",
    "What are the operating lease liabilities for COSTCO in 2021?",
]

with gr.Blocks(title="Multimodal RAG Pipeline", theme=gr.themes.Soft()) as app:

    gr.Markdown(
        "# Multimodal Hybrid RAG — Full Pipeline Demo\n"
        "Works for **any document type** — financial reports, research papers, "
        "charts, tables, academic documents.\n\n"
        "**Pipeline:** Query Expansion → BM25+Vector+RRF → "
        "DistilBERT CDE (F1=0.9274) → Qwen2.5-VL"
    )

    # ── Step 0: Upload (optional) ───────────────────────────────────
    with gr.Accordion("📁 Upload New Document into System ", open=False):
        gr.Markdown(
            "Upload an image or PDF. It will be **embedded and stored in Qdrant**, "
            "then becomes searchable using the query box below.\n"
            "This is part of the RAG pipeline — not a bypass."
        )
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Upload an Image")
                img_input   = gr.Image(
                    label="Upload image (chart, table, graph, screenshot)",
                    type="pil", height=200,
                    sources=["upload", "clipboard"],
                )
                img_docname = gr.Textbox(
                    label="Document name",
                    placeholder="e.g. SCHOOL_BARCHART_2024 or RESEARCH_FIGURE_1",
                )
                img_btn    = gr.Button("Embed & Index Image into Qdrant",
                                       variant="secondary")
                img_status = gr.Textbox(label="Image ingestion status",
                                        interactive=False, lines=5)
                img_btn.click(fn=ingest_image_to_qdrant,
                              inputs=[img_input, img_docname],
                              outputs=img_status)

            with gr.Column():
                gr.Markdown("#### Upload a PDF")
                pdf_input   = gr.File(
                    label="Upload PDF document",
                    file_types=[".pdf"],
                )
                pdf_docname = gr.Textbox(
                    label="Document name",
                    placeholder="e.g. APPLE_2023_10K or RESEARCH_PAPER",
                )
                pdf_btn    = gr.Button("Parse, Embed & Index PDF into Qdrant",
                                       variant="secondary")
                pdf_status = gr.Textbox(label="PDF ingestion status",
                                        interactive=False, lines=5)
                pdf_btn.click(fn=ingest_pdf_to_qdrant,
                              inputs=[pdf_input, pdf_docname],
                              outputs=pdf_status)

    gr.Markdown("---")

    # ── Query ───────────────────────────────────────────────────────
    gr.Markdown("## Ask a Question")
    gr.Markdown(
        "Type any question. The system retrieves relevant chunks from Qdrant "
        "(including any documents you just uploaded above), validates them with "
        "the CDE model, and generates an answer."
    )

    with gr.Row():
        with gr.Column(scale=3):
            qbox = gr.Textbox(
                label="Your question (any topic — financial, academic, research, etc.)",
                placeholder="e.g. What are the total assets for COSTCO? OR What does this chart show?",
                lines=2,
            )
            gr.Examples(examples=SAMPLES, inputs=qbox, label="Sample queries")

        with gr.Column(scale=1):
            mode = gr.Radio(
                choices=["Multimodal", "Text only"],
                value="Multimodal",
                label="Generation mode",
                info="Multimodal: model reads actual images. Text only: uses descriptions.",
            )
            run_btn    = gr.Button("Run Full RAG Pipeline",
                                   variant="primary", size="lg")
            status_box = gr.Textbox(label="Status", interactive=False, lines=1)

    gr.Markdown("---\n## Pipeline Intermediate Outputs")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Step 1 — Query Expansion")
            s1_out = gr.Markdown("_Sub-queries appear here._")
        with gr.Column():
            gr.Markdown("### Step 2 — Retrieval")
            s2_out = gr.Markdown("_BM25 + Vector + RRF scores appear here._")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Step 3 — CDE Validation (DistilBERT F1=0.9274)")
            s3_out = gr.Markdown("_Chunk confidence scores appear here._")
        with gr.Column():
            gr.Markdown("### Step 4 — Retrieved Images from Knowledge Base")
            gallery = gr.Gallery(
                label="Retrieved images",
                columns=3, height=260,
                object_fit="contain", show_label=False,
            )

    gr.Markdown("---\n## Final Answer")
    answer_box = gr.Textbox(
        label="Generated Answer",
        lines=7, interactive=False,
        placeholder="Answer with citations appears here...",
    )
    citations_out = gr.Markdown("_Sources appear here._")

    run_btn.click(
        fn=run_pipeline,
        inputs=[qbox, mode],
        outputs=[s1_out, s2_out, s3_out, gallery,
                 answer_box, citations_out, status_box],
    )

    gr.Markdown(
        "---\n"
       
    )


if __name__ == "__main__":
    print("Loading retrieval models...")
    get_models()
    print(f"LM Studio: {'ONLINE' if check_connection() else 'OFFLINE'}")
    print("Starting at http://localhost:7860")
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)



    #  "_Architecture: Query Expansion → Hybrid BM25+Vector+RRF → "
    #     "DistilBERT CDE (F1=0.9274) → Qwen2.5-VL-7B via LM Studio_"