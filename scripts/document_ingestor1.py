"""
document_ingestor.py
------------------
Ingest PDF documents and images into Qdrant collections.
Works with files from ANY folder location.
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

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import config
except ImportError:
    # Try relative import
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# Configuration
CHUNK_SIZE = 300    # words per text chunk
CHUNK_OVERLAP = 50  # word overlap between chunks
BATCH_SIZE = 16

# Handle image directory - where extracted images will be stored
if hasattr(config, 'IMAGES_DIR'):
    IMAGES_DIR = Path(config.IMAGES_DIR)
elif hasattr(config, 'IMAGE_DIR'):
    IMAGES_DIR = Path(config.IMAGE_DIR)
else:
    # Fallback to default path
    IMAGES_DIR = Path("data/raw/images")

# Create images directory if it doesn't exist
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ #
# PDF parsing - works with ANY file path
# ------------------------------------------------------------------ #

def parse_pdf(pdf_path: str) -> list[dict]:
    """
    Parse PDF using pymupdf (fitz) - works with ANY valid file path.
    Returns list of page dicts:
      {"page_num": int, "text": str, "images": [PIL.Image, ...]}
    """
    # Check if file exists
    if not os.path.exists(pdf_path):
        log.error(f"PDF file does not exist: {pdf_path}")
        return []
    
    log.info(f"Parsing PDF: {pdf_path}")
    
    try:
        import fitz
        doc = fitz.open(pdf_path)
        pages = []
        
        for i, page in enumerate(doc, start=1):
            # Extract text
            text = page.get_text("text") or ""
            
            # Render page as image
            mat = fitz.Matrix(1.5, 1.5)
            pix = page.get_pixmap(matrix=mat)
            pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            pages.append({
                "page_num": i, 
                "text": text, 
                "images": [pil]
            })
            
            if i % 10 == 0:
                log.info(f"  Parsed {i} pages...")
        
        doc.close()
        log.info(f"PDF parsed successfully: {len(pages)} pages")
        return pages
        
    except ImportError:
        log.error("pymupdf not installed. Run: pip install pymupdf")
        return []
    except Exception as e:
        log.error(f"PDF parsing failed: {e}")
        return []


# ------------------------------------------------------------------ #
# Text chunking
# ------------------------------------------------------------------ #

def chunk_text(text: str, page_num: int, doc_name: str,
               size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """Split page text into overlapping word-window chunks."""
    words = text.split()
    chunks = []
    i = 0
    chunk_idx = 0
    
    while i < len(words):
        chunk_words = words[i:i + size]
        chunk_text = " ".join(chunk_words).strip()
        
        if len(chunk_text) > 50:  # skip tiny chunks
            chunks.append({
                "quote_id": f"{doc_name}_p{page_num}_c{chunk_idx}",
                "doc_name": doc_name,
                "page_id": page_num,
                "chunk_idx": chunk_idx,
                "modality": "text",
                "text": chunk_text,
                "image_path": None,
            })
            chunk_idx += 1
        
        i += size - overlap
    
    return chunks


# ------------------------------------------------------------------ #
# Image handling - save extracted images to disk
# ------------------------------------------------------------------ #

def save_page_image(pil_img: Image.Image, doc_name: str, page_num: int) -> str:
    """Save PIL image to disk and return relative path."""
    filename = f"{doc_name}_page{page_num}.jpg"
    full_path = IMAGES_DIR / filename
    
    # Convert to RGB if needed
    if pil_img.mode in ("RGBA", "P"):
        pil_img = pil_img.convert("RGB")
    
    # Save image
    pil_img.save(str(full_path), "JPEG", quality=85)
    log.debug(f"Saved image: {full_path}")
    
    return f"images/{filename}"


# ------------------------------------------------------------------ #
# Get current max point IDs from Qdrant
# ------------------------------------------------------------------ #

def get_max_id(client: QdrantClient, collection_name: str) -> int:
    """Get the current maximum point ID to avoid collisions."""
    try:
        info = client.get_collection(collection_name)
        count = info.points_count
        return count + 1
    except Exception:
        return 1


# ------------------------------------------------------------------ #
# Main PDF ingestion - works with ANY file path
# ------------------------------------------------------------------ #

def ingest_pdf(pdf_path: str, doc_name: str = None):
    """
    Ingest a PDF document into Qdrant.
    
    Args:
        pdf_path: Path to the PDF file (can be ANY valid path)
        doc_name: Document name (optional, will be derived from filename if not provided)
    
    Returns:
        tuple: (success: bool, message: str)
    """
    # Validate input file exists
    if not os.path.exists(pdf_path):
        log.error(f"PDF not found: {pdf_path}")
        return False, f"PDF not found: {pdf_path}"

    # Generate document name if not provided
    if not doc_name:
        doc_name = Path(pdf_path).stem.replace(" ", "_").upper()
    
    log.info(f"Ingesting: {pdf_path} → doc_name={doc_name}")

    try:
        # Connect to Qdrant
        client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
        log.info(f"Connected to Qdrant at {config.QDRANT_HOST}:{config.QDRANT_PORT}")

        # Check if collections exist
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if config.TEXT_COLLECTION not in collection_names:
            log.warning(f"Collection {config.TEXT_COLLECTION} does not exist. Creating...")
            from vector_store import VectorStore
            vs = VectorStore()
            vs.setup_collections()

        # Load embedding models
        log.info("Loading embedding models...")
        text_model = SentenceTransformer(config.TEXT_MODEL, device=config.DEVICE)
        clip_model = CLIPModel.from_pretrained(config.IMAGE_MODEL).to(config.DEVICE)
        clip_processor = CLIPProcessor.from_pretrained(config.IMAGE_MODEL)
        clip_model.eval()

        # Parse PDF - this works with ANY valid file path
        pages = parse_pdf(pdf_path)
        
        if not pages:
            return False, "No pages could be extracted from PDF"

        # Collect all chunks
        text_chunks = []
        image_chunks = []

        for page in pages:
            # Text chunks
            if page["text"].strip():
                chunks = chunk_text(page["text"], page["page_num"], doc_name)
                text_chunks.extend(chunks)

            # Image chunks - save page image
            for pil_img in page["images"]:
                rel_path = save_page_image(pil_img, doc_name, page["page_num"])
                image_chunks.append({
                    "quote_id": f"{doc_name}_img_p{page['page_num']}",
                    "doc_name": doc_name,
                    "page_id": page["page_num"],
                    "modality": "image",
                    "text": f"Page {page['page_num']} image from {doc_name}",
                    "image_path": rel_path,
                })

        log.info(f"Chunks created: {len(text_chunks)} text, {len(image_chunks)} images")

        # Get starting point IDs
        text_start = get_max_id(client, config.TEXT_COLLECTION)
        image_start = get_max_id(client, config.IMAGE_COLLECTION)

        # Index text chunks
        if text_chunks:
            log.info(f"Embedding and indexing {len(text_chunks)} text chunks...")
            point_id = text_start
            
            for i in range(0, len(text_chunks), BATCH_SIZE):
                batch = text_chunks[i:i + BATCH_SIZE]
                texts = [c["text"] for c in batch]
                
                # Generate embeddings
                embs = text_model.encode(texts, show_progress_bar=False).tolist()
                
                # Create points
                points = []
                for j, (chunk, emb) in enumerate(zip(batch, embs)):
                    points.append(qmodels.PointStruct(
                        id=point_id,
                        vector=emb,
                        payload={
                            "quote_id": chunk["quote_id"],
                            "doc_name": chunk["doc_name"],
                            "page_id": chunk["page_id"],
                            "chunk_idx": chunk["chunk_idx"],
                            "modality": "text",
                            "text": chunk["text"],
                            "image_path": None,
                        }
                    ))
                    point_id += 1
                
                # Upload to Qdrant
                client.upsert(collection_name=config.TEXT_COLLECTION, points=points)
                log.info(f"  Indexed text batch {i}-{i + len(batch)}")

        # Index image chunks
        if image_chunks:
            log.info(f"Embedding and indexing {len(image_chunks)} image chunks...")
            point_id = image_start
            
            for chunk in image_chunks:
                # Get full path to the saved image
                full_path = Path(config.RAW_DATA_DIR) / chunk["image_path"]
                
                if not full_path.exists():
                    log.warning(f"Image not found: {full_path}")
                    continue
                
                try:
                    # Load image
                    pil = Image.open(str(full_path)).convert("RGB")
                    
                    # Generate CLIP embedding
                    inputs = clip_processor(images=[pil], return_tensors="pt").to(config.DEVICE)
                    with torch.no_grad():
                        feats = clip_model.get_image_features(**inputs)
                        feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
                        emb = feats.cpu().numpy().tolist()[0]
                    
                    # Upload to Qdrant
                    client.upsert(
                        collection_name=config.IMAGE_COLLECTION,
                        points=[qmodels.PointStruct(
                            id=point_id,
                            vector=emb,
                            payload={
                                "quote_id": chunk["quote_id"],
                                "doc_name": chunk["doc_name"],
                                "page_id": chunk["page_id"],
                                "modality": "image",
                                "text": chunk["text"],
                                "image_path": chunk["image_path"],
                            }
                        )]
                    )
                    point_id += 1
                    
                except Exception as e:
                    log.warning(f"Failed to process image {full_path}: {e}")
                    continue

        # Save manifest (optional - for tracking)
        manifest_path = Path(config.RAW_DATA_DIR) / f"{doc_name}_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump({
                "doc_name": doc_name,
                "pdf_path": str(pdf_path),
                "n_text": len(text_chunks),
                "n_images": len(image_chunks),
                "text_start_id": text_start,
                "image_start_id": image_start,
            }, f, indent=2)

        log.info(f"✅ Done. Indexed {len(text_chunks)} text + {len(image_chunks)} images for '{doc_name}'")
        
        return True, f"Successfully indexed '{doc_name}' with {len(text_chunks)} text chunks and {len(image_chunks)} images"

    except Exception as e:
        log.error(f"Error ingesting PDF: {e}")
        import traceback
        traceback.print_exc()
        return False, f"Error: {str(e)}"


# ------------------------------------------------------------------ #
# Single image ingestion
# ------------------------------------------------------------------ #
def ingest_image(pil_img: Image.Image, doc_name: str, page_id: int = 1) -> dict:
    """
    Embed a single PIL image with CLIP and upsert into image_collection.
    Also create a text chunk from the doc_name for BM25 searchability.
    Returns dict with status and info.
    """
    try:
        # Initialize
        client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
        text_model = SentenceTransformer(config.TEXT_MODEL, device=config.DEVICE)
        clip_model = CLIPModel.from_pretrained(config.IMAGE_MODEL).to(config.DEVICE)
        clip_proc = CLIPProcessor.from_pretrained(config.IMAGE_MODEL)
        clip_model.eval()

        # Save image to disk
        img_filename = f"{doc_name}_uploaded_p{page_id}.jpg"
        img_path = IMAGES_DIR / img_filename
        if pil_img.mode in ("RGBA", "P"):
            pil_img = pil_img.convert("RGB")
        pil_img.save(str(img_path), "JPEG", quality=90)
        rel_path = f"images/{img_filename}"

        # Get next IDs
        text_start = get_max_id(client, config.TEXT_COLLECTION)
        image_start = get_max_id(client, config.IMAGE_COLLECTION)

        # 1. Text chunk - doc_name as searchable text
        text_chunk = f"Document: {doc_name}. Uploaded image from page {page_id}."
        text_emb = text_model.encode([text_chunk])[0].tolist()
        client.upsert(
            collection_name=config.TEXT_COLLECTION,
            points=[qmodels.PointStruct(
                id=text_start,
                vector=text_emb,
                payload={
                    "quote_id": f"{doc_name}_text_p{page_id}",
                    "doc_name": doc_name,
                    "page_id": page_id,
                    "modality": "text",
                    "text": text_chunk,
                    "image_path": rel_path,
                }
            )]
        )

        # 2. Image embedding with CLIP
        inputs = clip_proc(images=[pil_img], return_tensors="pt").to(config.DEVICE)
        with torch.no_grad():
            outputs = clip_model.get_image_features(**inputs)

            # Extract the actual feature tensor (for older transformers versions)
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                feats = outputs.pooler_output
            elif hasattr(outputs, 'last_hidden_state'):
                feats = outputs.last_hidden_state[:, 0, :]
            else:
                feats = outputs

            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
            img_emb = feats.cpu().numpy().tolist()[0]

        # Upload to Qdrant
        client.upsert(
            collection_name=config.IMAGE_COLLECTION,
            points=[qmodels.PointStruct(
                id=image_start,
                vector=img_emb,
                payload={
                    "quote_id": f"{doc_name}_image_p{page_id}",
                    "doc_name": doc_name,
                    "page_id": page_id,
                    "modality": "image",
                    "text": f"Image from {doc_name} page {page_id}",
                    "image_path": rel_path,
                }
            )]
        )

        log.info("Ingested image: doc=%s page=%d → text_id=%d image_id=%d",
                 doc_name, page_id, text_start, image_start)

        return {
            "success": True,
            "message": f"Image '{doc_name}' added successfully",
            "doc_name": doc_name,
            "page_id": page_id,
            "text_id": text_start,
            "image_id": image_start,
            "rel_path": rel_path,
        }

    except Exception as e:
        log.error(f"Error ingesting image: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "message": f"Error: {str(e)}",
            "doc_name": doc_name,
        }

# --------scripts/document_ingestor.py---------------------------------------------------------- #
# Command line interface
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into Qdrant")
    parser.add_argument("--pdf", help="Path to PDF file (can be any location)")
    parser.add_argument("--doc_name", default=None, help="Document name (default: PDF filename)")
    
    args = parser.parse_args()
    
    if args.pdf:
        print(f"\n📄 Processing PDF from: {args.pdf}")
        success, message = ingest_pdf(args.pdf, args.doc_name)
        print(f"\n{message}")
    else:
        print("\nPlease provide --pdf path")
        print("Example: python document_ingestor.py --pdf C:\\Users\\YourName\\Downloads\\document.pdf")