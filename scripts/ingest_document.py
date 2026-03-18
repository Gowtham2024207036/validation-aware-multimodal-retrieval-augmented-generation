"""
ingest_document.py
------------------
Ingest a NEW PDF document into the existing Qdrant collections.

Pipeline:
  1. Parse PDF → extract text blocks and images per page
  2. Chunk text into passages (~300 tokens each)
  3. Generate image descriptions using Qwen2.5-VL (optional)
  4. Embed text chunks with SentenceTransformer (768-dim)
  5. Embed images with CLIP (512-dim)
  6. Upsert into Qdrant text_collection and image_collection
  7. Save metadata (doc_name, page_id, chunk_id) for citations

Install extras:
    pip install pdfplumber pillow

Usage:
    python scripts/ingest_document.py --pdf path/to/report.pdf --doc_name APPLE_2023_10K
    python scripts/ingest_document.py --pdf path/to/report.pdf  # auto doc_name from filename
"""

import os
import sys
import json
import re
import argparse
import logging
from pathlib import Path

import torch
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

CHUNK_SIZE    = 300    # words per text chunk
CHUNK_OVERLAP = 50     # word overlap between chunks
IMAGES_DIR    = Path("data/raw/images")
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE    = 16


# ------------------------------------------------------------------ #
# PDF parsing
# ------------------------------------------------------------------ #

def parse_pdf(pdf_path: str) -> list[dict]:
    """
    Parse PDF using pymupdf (fitz) — better for multilingual PDFs.
    Falls back to pdfplumber if pymupdf not available.
    Returns list of page dicts:
      {"page_num": int, "text": str, "images": [PIL.Image, ...]}
    """
    # Try pymupdf first — handles Tamil, Hindi, and other scripts well
    try:
        import fitz
        doc    = fitz.open(pdf_path)
        pages  = []
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text") or ""
            # Render page as image
            mat  = fitz.Matrix(1.5, 1.5)
            pix  = page.get_pixmap(matrix=mat)
            pil  = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            pages.append({"page_num": i, "text": text, "images": [pil]})
            if i % 10 == 0:
                log.info("  Parsed %d pages...", i)
        doc.close()
        log.info("PDF parsed with pymupdf: %d pages", len(pages))
        return pages
    except ImportError:
        pass
    except Exception as e:
        log.warning("pymupdf failed (%s), trying pdfplumber...", e)

    # Fallback to pdfplumber
    try:
        import pdfplumber
    except ImportError:
        log.error("No PDF library found. Run: pip install pymupdf")
        sys.exit(1)

    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text   = page.extract_text() or ""
            images = []
            try:
                pil = page.to_image(resolution=150).original
                images.append(pil)
            except Exception:
                pass
            pages.append({"page_num": i, "text": text, "images": images})
            if i % 10 == 0:
                log.info("  Parsed %d pages...", i)
    log.info("PDF parsed with pdfplumber: %d pages", len(pages))
    return pages


# ------------------------------------------------------------------ #
# Text chunking
# ------------------------------------------------------------------ #

def chunk_text(text: str, page_num: int, doc_name: str,
               size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """Split page text into overlapping word-window chunks."""
    words  = text.split()
    chunks = []
    i      = 0
    chunk_idx = 0
    while i < len(words):
        chunk_words = words[i : i + size]
        chunk_text  = " ".join(chunk_words).strip()
        if len(chunk_text) > 50:   # skip tiny chunks
            chunks.append({
                "quote_id":   f"{doc_name}_p{page_num}_c{chunk_idx}",
                "doc_name":   doc_name,
                "page_id":    page_num,
                "chunk_idx":  chunk_idx,
                "modality":   "text",
                "text":       chunk_text,
                "image_path": None,
            })
            chunk_idx += 1
        i += size - overlap
    return chunks


# ------------------------------------------------------------------ #
# Image handling
# ------------------------------------------------------------------ #

def save_page_image(pil_img: Image.Image, doc_name: str, page_num: int) -> str:
    """Save PIL image to disk and return relative path."""
    filename  = f"{doc_name}_page{page_num}.jpg"
    full_path = IMAGES_DIR / filename
    if pil_img.mode in ("RGBA", "P"):
        pil_img = pil_img.convert("RGB")
    pil_img.save(str(full_path), "JPEG", quality=85)
    return f"images/{filename}"


# ------------------------------------------------------------------ #
# Get current max point IDs from Qdrant
# ------------------------------------------------------------------ #

def get_max_id(client: QdrantClient, collection_name: str) -> int:
    """Get the current maximum point ID to avoid collisions."""
    try:
        info  = client.get_collection(collection_name)
        count = info.points_count
        return count + 1
    except Exception:
        return 1


# ------------------------------------------------------------------ #
# Main ingestion
# ------------------------------------------------------------------ #

def ingest(pdf_path: str, doc_name: str = None):
    if not os.path.exists(pdf_path):
        log.error("PDF not found: %s", pdf_path)
        sys.exit(1)

    if not doc_name:
        doc_name = Path(pdf_path).stem.replace(" ", "_").upper()
    log.info("Ingesting: %s → doc_name=%s", pdf_path, doc_name)

    # Connect
    client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
    log.info("Connected to Qdrant at %s:%s", config.QDRANT_HOST, config.QDRANT_PORT)

    # Load models
    log.info("Loading embedding models...")
    text_model     = SentenceTransformer(config.TEXT_MODEL,  device=config.DEVICE)
    clip_model     = CLIPModel.from_pretrained(config.IMAGE_MODEL).to(config.DEVICE)
    clip_processor = CLIPProcessor.from_pretrained(config.IMAGE_MODEL)
    clip_model.eval()

    # Parse PDF
    pages = parse_pdf(pdf_path)

    # Collect all chunks
    text_chunks  = []
    image_chunks = []

    for page in pages:
        # Text chunks
        chunks = chunk_text(page["text"], page["page_num"], doc_name)
        text_chunks.extend(chunks)

        # Image chunks — save page image
        for pil_img in page["images"]:
            rel_path = save_page_image(pil_img, doc_name, page["page_num"])
            image_chunks.append({
                "quote_id":   f"{doc_name}_img_p{page['page_num']}",
                "doc_name":   doc_name,
                "page_id":    page["page_num"],
                "modality":   "image",
                "text":       f"Page {page['page_num']} image from {doc_name}",
                "image_path": rel_path,
            })

    log.info("Chunks: %d text, %d images", len(text_chunks), len(image_chunks))

    # Get starting point IDs
    text_start  = get_max_id(client, config.TEXT_COLLECTION)
    image_start = get_max_id(client, config.IMAGE_COLLECTION)

    # ── Index text chunks ──────────────────────────────────────────
    log.info("Embedding and indexing text chunks...")
    point_id = text_start
    for i in range(0, len(text_chunks), BATCH_SIZE):
        batch = text_chunks[i : i + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        embs  = text_model.encode(texts, show_progress_bar=False).tolist()
        points = []
        for j, (chunk, emb) in enumerate(zip(batch, embs)):
            points.append(qmodels.PointStruct(
                id      = point_id,
                vector  = emb,
                payload = {
                    "quote_id":   chunk["quote_id"],
                    "doc_name":   chunk["doc_name"],
                    "page_id":    chunk["page_id"],
                    "chunk_idx":  chunk["chunk_idx"],
                    "modality":   "text",
                    "text":       chunk["text"],
                    "image_path": None,
                }
            ))
            point_id += 1
        client.upsert(collection_name=config.TEXT_COLLECTION, points=points)
        log.info("  Text: indexed batch %d-%d", i, i + len(batch))

    # ── Index image chunks ─────────────────────────────────────────
    log.info("Embedding and indexing image chunks...")
    point_id = image_start
    for chunk in image_chunks:
        full_path = Path("data/raw") / chunk["image_path"]
        if not full_path.exists():
            continue
        try:
            pil = Image.open(str(full_path)).convert("RGB")
        except Exception as e:
            log.warning("Cannot open %s: %s", full_path, e)
            continue
        inputs = clip_processor(images=[pil], return_tensors="pt").to(config.DEVICE)
        with torch.no_grad():
            feats = clip_model.get_image_features(**inputs)
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
            emb   = feats.cpu().numpy().tolist()[0]
        client.upsert(
            collection_name=config.IMAGE_COLLECTION,
            points=[qmodels.PointStruct(
                id      = point_id,
                vector  = emb,
                payload = {
                    "quote_id":   chunk["quote_id"],
                    "doc_name":   chunk["doc_name"],
                    "page_id":    chunk["page_id"],
                    "modality":   "image",
                    "text":       chunk["text"],
                    "image_path": chunk["image_path"],
                }
            )]
        )
        point_id += 1

    # Save manifest
    manifest_path = Path("data/raw") / f"{doc_name}_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "doc_name":     doc_name,
            "pdf_path":     str(pdf_path),
            "n_text":       len(text_chunks),
            "n_images":     len(image_chunks),
            "text_start_id": text_start,
            "image_start_id": image_start,
        }, f, indent=2)

    log.info("Done. Indexed %d text + %d image chunks for '%s'",
             len(text_chunks), len(image_chunks), doc_name)
    log.info("Manifest saved: %s", manifest_path)




# ------------------------------------------------------------------ #
# Ingest a single uploaded image into Qdrant
# ------------------------------------------------------------------ #

def ingest_image(
    pil_img:  "Image.Image",
    doc_name: str,
    page_id:  int = 1,
) -> dict:
    """
    Embed a single PIL image with CLIP and upsert into image_collection.
    Also create a text chunk from the doc_name for BM25 searchability.
    Returns {"text_id": int, "image_id": int, "doc_name": str}
    """
    import torch
    client     = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
    text_model = SentenceTransformer(config.TEXT_MODEL, device=config.DEVICE)
    clip_model = CLIPModel.from_pretrained(config.IMAGE_MODEL).to(config.DEVICE)
    clip_proc  = CLIPProcessor.from_pretrained(config.IMAGE_MODEL)
    clip_model.eval()

    # Save image to disk
    img_filename = f"{doc_name}_uploaded_p{page_id}.jpg"
    img_path     = IMAGES_DIR / img_filename
    if pil_img.mode in ("RGBA", "P"):
        pil_img = pil_img.convert("RGB")
    pil_img.save(str(img_path), "JPEG", quality=90)
    rel_path = f"images/{img_filename}"

    # Get next IDs
    text_start  = get_max_id(client, config.TEXT_COLLECTION)
    image_start = get_max_id(client, config.IMAGE_COLLECTION)

    # 1. Text chunk — doc_name as searchable text so BM25 can find it
    text_chunk = f"Document: {doc_name}. Uploaded image from page {page_id}."
    text_emb   = text_model.encode([text_chunk])[0].tolist()
    client.upsert(
        collection_name=config.TEXT_COLLECTION,
        points=[qmodels.PointStruct(
            id      = text_start,
            vector  = text_emb,
            payload = {
                "quote_id":   f"{doc_name}_text_p{page_id}",
                "doc_name":   doc_name,
                "page_id":    page_id,
                "modality":   "text",
                "text":       text_chunk,
                "image_path": rel_path,
            }
        )]
    )

    # 2. Image embedding with CLIP
    inputs = clip_proc(images=[pil_img], return_tensors="pt").to(config.DEVICE)
    with torch.no_grad():
        feats = clip_model.get_image_features(**inputs)
        feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
        img_emb = feats.cpu().numpy().tolist()[0]

    client.upsert(
        collection_name=config.IMAGE_COLLECTION,
        points=[qmodels.PointStruct(
            id      = image_start,
            vector  = img_emb,
            payload = {
                "quote_id":   f"{doc_name}_image_p{page_id}",
                "doc_name":   doc_name,
                "page_id":    page_id,
                "modality":   "image",
                "text":       f"Image from {doc_name} page {page_id}",
                "image_path": rel_path,
            }
        )]
    )

    log.info("Ingested image: doc=%s page=%d → text_id=%d image_id=%d",
             doc_name, page_id, text_start, image_start)

    return {
        "doc_name":  doc_name,
        "page_id":   page_id,
        "text_id":   text_start,
        "image_id":  image_start,
        "rel_path":  rel_path,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf",      required=True, help="Path to PDF file")
    parser.add_argument("--doc_name", default=None,  help="Document name (default: PDF filename)")
    args = parser.parse_args()
    ingest(args.pdf, args.doc_name)