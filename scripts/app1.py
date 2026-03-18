"""
app.py  —  Multimodal RAG Demo (Aligned with Architecture)
Shows ALL intermediate steps: Query Expansion → Hybrid Retrieval → CDE → Generation
Run: python scripts/app.py  →  http://localhost:7860
"""
import os, sys, json, threading
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gradio as gr
from base_retriever import SharedModels
from lmstudio_client import check_connection, generate_text_only, generate_with_images, LM_STUDIO_BASE_URL
from prompt_builder import SYSTEM_PROMPT, build_text_only_prompt, build_multimodal_prompt
import arch6_full_proposed as best_arch

IMAGES_ROOT = Path("data/raw")
_models = None
_models_lock = threading.Lock()

def get_models():
    global _models
    with _models_lock:
        if _models is None:
            _models = SharedModels()
    return _models

def fmt_step1(sub_queries):
    if not sub_queries: return "_No sub-queries._"
    lines = ["**Query Expansion — Generated Sub-Queries:**\n"]
    for i,q in enumerate(sub_queries,1): lines.append(f"{i}. {q}")
    return "\n".join(lines)

def fmt_step2(text_hits, image_hits):
    lines = [f"**Retrieval** — {len(text_hits)} text + {len(image_hits)} image chunks\n"]
    for i,h in enumerate(text_hits,1):
        doc=h.get("doc_name","?"); page=h.get("page_id","?"); qid=h.get("quote_id","")
        rrf=h.get("rrf_score",h.get("score",0)); bm25=h.get("bm25_score"); vec=h.get("vector_score")
        txt=(h.get("text") or "")[:200]
        sc=[f"RRF:{rrf:.4f}"]
        if bm25: sc.append(f"BM25:{bm25:.1f}")
        if vec:  sc.append(f"Vec:{vec:.4f}")
        lines.append(f"\n**[Text {i}]** `{doc}` — Page {page} | {qid}\n_{' | '.join(sc)}_\n> {txt}...")
    for i,h in enumerate(image_hits,1):
        doc=h.get("doc_name","?"); page=h.get("page_id","?"); path=h.get("image_path","")
        rrf=h.get("rrf_score",0)
        lines.append(f"\n**[Image {i}]** `{doc}` — Page {page}\n_RRF:{rrf:.4f} | {path}_")
    return "\n".join(lines)

def fmt_step3(text_hits, image_hits):
    all_hits = [(h,"text") for h in text_hits]+[(h,"img") for h in image_hits]
    has_cde  = any(h.get("cde_confidence") is not None for h,_ in all_hits)
    if not has_cde: return "_CDE scores: using Architecture 6 shows these scores._"
    lines = ["**CDE Validation Scores** (DistilBERT F1=0.9274)\n"]
    lines.append("| # | Type | Document | Page | Relevance | Confidence | Kept |")
    lines.append("|---|------|----------|------|-----------|------------|------|")
    for i,(h,mod) in enumerate(all_hits,1):
        doc=h.get("doc_name","?")[:25]; page=h.get("page_id","?")
        rel=h.get("cde_relevance","—"); conf=h.get("cde_confidence","—")
        red="❌" if h.get("cde_redundant") else "✅"
        icon="📄" if mod=="text" else "🖼️"
        rels=f"{rel:.3f}" if isinstance(rel,float) else str(rel)
        confs=f"**{conf:.3f}**" if isinstance(conf,float) else str(conf)
        lines.append(f"| {i} | {icon} | {doc} | {page} | {rels} | {confs} | {red} |")
    return "\n".join(lines)

def fmt_citations(text_hits, image_hits):
    lines=["**Sources:**"]
    for i,h in enumerate(text_hits,1):
        lines.append(f"📄 [{i}] `{h.get('doc_name','?')}` — Page {h.get('page_id','?')} ({h.get('quote_id','')})")
    for i,h in enumerate(image_hits,1):
        lines.append(f"🖼️ [Img{i}] `{h.get('doc_name','?')}` — Page {h.get('page_id','?')} | {h.get('image_path','')}")
    return "\n".join(lines)

def ingest_pdf(file_obj, doc_name_input):
    if file_obj is None: return "No file uploaded."
    try:
        from ingest_document import ingest
        doc_name = doc_name_input.strip() or Path(file_obj.name).stem.replace(" ","_").upper()
        ingest(file_obj.name, doc_name)
        return f"✅ '{doc_name}' indexed. You can now query it above."
    except Exception as e:
        return f"❌ Ingestion failed: {e}"

def pil_save_temp(uploaded_image) -> str | None:
    """Save numpy image array to temp file, return path."""
    try:
        import tempfile
        from PIL import Image as PI
        pil = PI.fromarray(uploaded_image.astype("uint8")).convert("RGB")
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        pil.save(tmp.name, "JPEG", quality=90)
        return tmp.name
    except Exception as e:
        return None


def run_direct_image_qa(query: str, uploaded_image) -> tuple:
    """
    When user uploads an image: skip Qdrant entirely.
    Send image + question directly to Qwen2.5-VL.
    Returns same 7-tuple as run_pipeline.
    """
    if not check_connection():
        ans = f"⚠️ LM Studio offline. Start it at {LM_STUDIO_BASE_URL}"
        return ("_Skipped — direct image QA mode_",
                "_Skipped — no retrieval needed_",
                "_Skipped — no retrieval needed_",
                [(uploaded_image, "📤 Your uploaded image")],
                ans, "_No citations — direct image QA_",
                "Mode: Direct Image QA | LM Studio OFFLINE")

    tmp_path = pil_save_temp(uploaded_image)
    if not tmp_path:
        return ("_Error saving image_","_","_",
                [(uploaded_image,"Your image")],"❌ Could not process image.","","ERROR")

    system = (
        "You are an expert data analyst. "
        "The user has uploaded an image (chart, table, graph, or document). "
        "Analyse it carefully and answer the question based on what you see. "
        "Extract specific numbers, values, and trends visible in the image. "
        "Be precise and direct."
    )
    user_text = f"Question: {query}\n\nPlease examine the image carefully and answer."

    ans = generate_with_images(
        system_prompt = system,
        user_text     = user_text,
        image_paths   = [tmp_path],
        images_root   = "",
        max_tokens    = 1024,
        temperature   = 0.1,
    )

    s1 = "**Mode: Direct Image QA** _(uploaded image — Qdrant retrieval skipped)_"
    s2 = "_Not applicable — question answered directly from your uploaded image._"
    s3 = "_Not applicable — CDE validation only used for Qdrant-retrieved chunks._"
    cit = "_Source: Your uploaded image_"
    status = "Mode: Direct Image QA | Qwen2.5-VL reads your image directly"

    return s1, s2, s3, [(uploaded_image, "📤 Your uploaded image")], ans, cit, status


def run_pipeline(query, mode, uploaded_image):
    if not query.strip():
        e="_Run a query to see results._"
        return e,e,e,[],e,e,""

    # KEY DECISION: if user uploaded an image, answer from it directly
    # Do NOT search Qdrant for unrelated uploaded images
    if uploaded_image is not None:
        return run_direct_image_qa(query, uploaded_image)

    # No uploaded image — use full RAG pipeline with Qdrant
    models = get_models()
    try:
        result=best_arch.retrieve(query,models)
        text_hits=result.get("text",[]); image_hits=result.get("image",[]); sub_qs=result.get("sub_queries",[query])
    except Exception as e:
        err=f"❌ Retrieval error: {e}"
        return err,err,err,[],err,err,"ERROR"

    s1=fmt_step1(sub_qs); s2=fmt_step2(text_hits,image_hits); s3=fmt_step3(text_hits,image_hits)

    gallery=[]
    for h in image_hits:
        path=h.get("image_path",""); doc=h.get("doc_name",""); page=h.get("page_id","?")
        conf=h.get("cde_confidence"); cap=f"{doc} p.{page}"+(f" CDE:{conf:.2f}" if conf else "")
        full=IMAGES_ROOT/path if path else None
        gallery.append((str(full) if full and full.exists() else None, cap))

    if not check_connection():
        ans=f"⚠️ LM Studio offline. Start it at {LM_STUDIO_BASE_URL}"
        return s1,s2,s3,gallery,ans,fmt_citations(text_hits,image_hits),"Retrieval OK | LM Studio OFFLINE"

    try:
        if mode=="Multimodal":
            ut,ip=build_multimodal_prompt(query,text_hits,image_hits)
            ans=generate_with_images(SYSTEM_PROMPT,ut,ip,str(IMAGES_ROOT),1024,0.1)
        else:
            ut,_=build_text_only_prompt(query,text_hits,image_hits)
            ans=generate_text_only(SYSTEM_PROMPT,ut,1024,0.1)
    except Exception as e:
        ans=f"❌ Generation error: {e}"

    status=f"Arch 6 CDE | {len(sub_qs)} sub-queries | {len(text_hits)}T+{len(image_hits)}I | {mode}"
    return s1,s2,s3,gallery,ans,fmt_citations(text_hits,image_hits),status

SAMPLES=["What is the long-term debt to total liabilities for COSTCO in FY2021?",
         "What are the total assets for COSTCO as of August 2021?",
         "What is the net income for COSTCO in FY2021?",
         "What are the operating lease liabilities for COSTCO in 2021?",]

with gr.Blocks(title="Validation-Aware Multimodal RAG",theme=gr.themes.Soft()) as app:
    gr.Markdown("# Validation-Aware Multimodal RAG\nAll intermediate steps visible: Query Expansion → BM25+Vector+RRF → CDE Validation → Qwen2.5-VL")

    with gr.Row():
        with gr.Column(scale=3):
            qbox=gr.Textbox(label="Enter your question",placeholder="e.g. What is the long-term debt to total liabilities for COSTCO in FY2021?",lines=2)
            gr.Examples(examples=SAMPLES,inputs=qbox,label="Sample queries")
        with gr.Column(scale=1):
            mode=gr.Radio(choices=["Text only","Multimodal"],value="Text only",label="Mode")
            btn=gr.Button("▶ Run Full Pipeline",variant="primary",size="lg")
            status=gr.Textbox(label="Status",interactive=False,lines=1)

    with gr.Accordion("📎 Upload image for visual QA",open=False):
        img_up=gr.Image(label="Upload chart/table",type="numpy",height=180,sources=["upload","clipboard"])

    gr.Markdown("---\n## 🔍 Intermediate Pipeline Outputs")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Step 1 — Query Expansion")
            s1=gr.Markdown("_Results appear here after running._")
        with gr.Column():
            gr.Markdown("### Step 2 — Retrieval")
            s2=gr.Markdown("_BM25 + Vector + RRF scores appear here._")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Step 3 — CDE Validation (F1=0.9274)")
            s3=gr.Markdown("_Confidence scores per chunk appear here._")
        with gr.Column():
            gr.Markdown("### Step 4 — Retrieved Images")
            gal=gr.Gallery(label="Retrieved financial charts",columns=3,height=260,object_fit="contain",show_label=False)

    gr.Markdown("---\n## 💬 Final Answer")
    ans=gr.Textbox(label="LLM Answer",lines=7,interactive=False,placeholder="Answer with citations appears here...")
    cit=gr.Markdown("_Citations appear here._")

    with gr.Accordion("📄 Index New Document into System",open=False):
        gr.Markdown("Upload a PDF → indexed into Qdrant → queryable above.")
        with gr.Row():
            pdf_in=gr.File(label="Upload PDF",file_types=[".pdf"])
            dname=gr.Textbox(label="Document name",placeholder="e.g. APPLE_2023_10K")
        ibtn=gr.Button("Index Document",variant="secondary")
        istat=gr.Textbox(label="Ingestion status",interactive=False,lines=2)
        ibtn.click(fn=ingest_pdf,inputs=[pdf_in,dname],outputs=istat)

    btn.click(fn=run_pipeline,inputs=[qbox,mode,img_up],outputs=[s1,s2,s3,gal,ans,cit,status])

    gr.Markdown("---\n_Full Proposed Architecture: Query Expansion → Hybrid BM25+Vector+RRF → DistilBERT CDE (F1=0.9274) → Qwen2.5-VL-7B_")

if __name__=="__main__":
    print("Loading models..."); get_models()
    print(f"LM Studio: {'ONLINE' if check_connection() else 'OFFLINE'}")
    print("http://localhost:7860")
    app.launch(server_name="0.0.0.0",server_port=7860,share=False,show_error=True)