import os
import sys
import json
import logging
import torch
from tqdm import tqdm
from PIL import Image

# Ensure Python can find the 'config' module in your root folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel

# ---------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# Configuration & Paths
# ---------------------------------------------------------
# Always resolve paths relative to this script file, not the working directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE   = os.path.join(SCRIPT_DIR, "..", "data", "processed", "quotes_master.jsonl")
IMAGES_ROOT = os.path.join(SCRIPT_DIR, "..", "data", "raw")

BATCH_SIZE = 32  # Optimized for Quadro P5000


def initialize_qdrant():
    """Connects to Qdrant and recreates clean collections."""
    logger.info("Connecting to Qdrant at %s:%s...", config.QDRANT_HOST, config.QDRANT_PORT)
    client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)

    # Fix: recreate_collection() is removed in qdrant-client >= 1.7
    # Use delete + create instead
    for name, dim in [
        (config.TEXT_COLLECTION,  config.TEXT_VECTOR_SIZE),
        (config.IMAGE_COLLECTION, config.IMAGE_VECTOR_SIZE),
    ]:
        if client.collection_exists(name):
            logger.info("Deleting existing collection: %s", name)
            client.delete_collection(name)
        logger.info("Creating collection: %s (dim: %d)", name, dim)
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )

    return client


def read_data(data_file: str) -> tuple[list, list]:
    """Reads quotes_master.jsonl and splits into text and image rows."""
    text_rows  = []
    image_rows = []
    skipped    = 0

    logger.info("Reading data from %s...", data_file)
    with open(data_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            # Fix: handle malformed JSON gracefully
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("Skipping malformed JSON at line %d: %s", line_num, e)
                skipped += 1
                continue

            modality = row.get("modality")
            if modality == "text":
                text_rows.append(row)
            elif modality == "image":
                image_rows.append(row)
            else:
                logger.warning("Unknown modality %r at line %d — skipping", modality, line_num)
                skipped += 1

    logger.info(
        "Found %d text quotes and %d image quotes (%d lines skipped).",
        len(text_rows), len(image_rows), skipped,
    )
    return text_rows, image_rows


def index_text(client: QdrantClient, text_rows: list, text_model: SentenceTransformer) -> int:
    """Encodes and upserts all text rows. Returns count of indexed points."""
    logger.info("Starting text indexing (%d rows)...", len(text_rows))

    # Fix: use a dedicated counter so IDs stay unique even if rows are skipped
    point_id = 1
    indexed  = 0

    for i in tqdm(range(0, len(text_rows), BATCH_SIZE), desc="Indexing text"):
        batch = text_rows[i : i + BATCH_SIZE]
        texts = [row["text"] for row in batch]

        embeddings = text_model.encode(texts, show_progress_bar=False).tolist()

        points = []
        for j, row in enumerate(batch):
            points.append(models.PointStruct(
                id=point_id,
                vector=embeddings[j],
                payload={
                    "quote_id": row.get("quote_id"),
                    "modality": row.get("modality"),
                    "text":     row.get("text", ""),
                    "doc_name": row.get("doc_name"),
                },
            ))
            point_id += 1

        # Fix: catch Qdrant upsert errors per batch so one failure doesn't crash the run
        try:
            client.upsert(collection_name=config.TEXT_COLLECTION, points=points)
            indexed += len(points)
        except Exception as e:
            logger.error("Upsert failed for text batch at offset %d: %s", i, e)

    return indexed


def index_images(
    client: QdrantClient,
    image_rows: list,
    image_model: CLIPModel,
    image_processor: CLIPProcessor,
) -> tuple[int, int]:
    """Encodes and upserts all image rows. Returns (indexed_count, skipped_count)."""
    logger.info("Starting image indexing (%d rows)...", len(image_rows))

    # Fix: dedicated counter — not derived from batch offset, so always unique
    point_id = 1
    indexed  = 0
    missing  = 0

    for i in tqdm(range(0, len(image_rows), BATCH_SIZE), desc="Indexing images"):
        batch       = image_rows[i : i + BATCH_SIZE]
        valid_batch = []
        pil_images  = []

        for row in batch:
            # Fix: build absolute path so script works from any working directory
            img_path = os.path.join(IMAGES_ROOT, row.get("image_path", ""))

            if not os.path.exists(img_path):
                missing += 1
                continue

            try:
                img = Image.open(img_path).convert("RGB")
                pil_images.append(img)
                valid_batch.append(row)
            except Exception as e:
                logger.warning("Could not open image %s: %s", img_path, e)
                missing += 1

        if not pil_images:
            continue

        # Generate 512-dim visual embeddings via CLIP
        with torch.no_grad():
            inputs = image_processor(images=pil_images, return_tensors="pt").to(config.DEVICE)
            image_features = image_model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            embeddings = image_features.cpu().numpy().tolist()

        points = []
        for j, row in enumerate(valid_batch):
            points.append(models.PointStruct(
                id=point_id,
                vector=embeddings[j],
                payload={
                    "quote_id":   row.get("quote_id"),
                    "modality":   row.get("modality"),
                    "text":       row.get("text", ""),   # Fix: safe .get() with fallback
                    "image_path": row.get("image_path"),
                    "doc_name":   row.get("doc_name"),
                },
            ))
            point_id += 1

        # Fix: catch Qdrant upsert errors per batch
        try:
            client.upsert(collection_name=config.IMAGE_COLLECTION, points=points)
            indexed += len(points)
        except Exception as e:
            logger.error("Upsert failed for image batch at offset %d: %s", i, e)

    return indexed, missing


def index_dataset():
    client = initialize_qdrant()

    text_rows, image_rows = read_data(DATA_FILE)

    # --- Text indexing ---
    logger.info("Loading text model onto %s...", config.DEVICE.upper())
    text_model = SentenceTransformer(config.TEXT_MODEL, device=config.DEVICE)

    text_indexed = index_text(client, text_rows, text_model)

    # Fix: free GPU memory before loading CLIP to avoid VRAM pressure
    del text_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Text model unloaded from GPU.")

    # --- Image indexing ---
    logger.info("Loading image model onto %s...", config.DEVICE.upper())
    image_model     = CLIPModel.from_pretrained(config.IMAGE_MODEL).to(config.DEVICE)
    image_processor = CLIPProcessor.from_pretrained(config.IMAGE_MODEL)

    image_indexed, image_missing = index_images(client, image_rows, image_model, image_processor)

    # Fix: final summary so you can verify the run completed correctly
    logger.info("=" * 50)
    logger.info("Indexing complete.")
    logger.info("  Text points indexed  : %d / %d", text_indexed,  len(text_rows))
    logger.info("  Image points indexed : %d / %d", image_indexed, len(image_rows))
    if image_missing > 0:
        logger.warning(
            "  Images skipped (missing/corrupt): %d — check data/raw/images/",
            image_missing,
        )
    logger.info("=" * 50)


if __name__ == "__main__":
    index_dataset()