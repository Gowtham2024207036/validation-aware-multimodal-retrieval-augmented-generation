"""
bulk_update_descriptions.py
----------------------------
Updates ALL 1720 image points in Qdrant with rich text descriptions
from the original MMDocRAG dataset files.

WHY THIS IS NEEDED:
  When images were indexed by index_dataset.py, the 'text' field stored
  in Qdrant was minimal (just quote_id or a short label).
  BM25 can't find images by content because there's nothing to match.

  The MMDocRAG source files (dev_20.jsonl, evaluation_20.jsonl, train.jsonl)
  contain detailed img_description fields like:
    "The table shows a breakdown of long-term debt for 2021 and 2020..."
  
  This script reads those descriptions and updates the Qdrant payload
  so BM25 can find images by their actual visual content.

RESULT:
  Before: query "gas price 2022" → no image results
  After:  query "gas price 2022" → finds the AAA gas price chart

Usage:
    python scripts/bulk_update_descriptions.py
    python scripts/bulk_update_descriptions.py --dry_run   # preview only
    python scripts/bulk_update_descriptions.py --batch_size 50
"""

import os
import sys
import re
import json
import argparse
import logging
from pathlib import Path
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")

# Source files to read descriptions from (in priority order)
SOURCE_FILES = [
    "dev_20.jsonl",
    "dev_15.jsonl",
    "evaluation_20.jsonl",
    "evaluation_15.jsonl",
    "train.jsonl",
]


# ------------------------------------------------------------------ #
# Step 1: Build description map from source files
# ------------------------------------------------------------------ #

def build_description_map() -> dict[str, str]:
    """
    Read all MMDocRAG source files and build a map:
      image_path → rich_description
    Also builds:
      quote_id   → rich_description
    """
    desc_by_path  = {}   # "images/COSTCO_2021_10K_image17.jpg" -> description
    desc_by_qid   = {}   # "image7" -> description
    desc_by_docimg = {}  # ("COSTCO_2021_10K", "image17") -> description

    files_read   = 0
    records_read = 0

    for fname in SOURCE_FILES:
        fpath = RAW_DIR / fname
        if not fpath.exists():
            log.warning("Not found: %s", fpath)
            continue

        log.info("Reading %s ...", fname)
        files_read += 1

        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                doc_name = rec.get("doc_name", "")

                # Format B: flat format (dev/eval files)
                for img_q in rec.get("img_quotes", []):
                    qid   = img_q.get("quote_id", "")        # e.g. "image7"
                    path  = img_q.get("img_path", "")        # e.g. "images/COSTCO_2021_10K_image17.jpg"
                    desc  = img_q.get("img_description", "").strip()

                    if not desc:
                        continue

                    # Enrich description with doc_name for better BM25 matching
                    enriched = f"{doc_name} {desc}" if doc_name else desc

                    if path:
                        # Normalise path — remove leading slash or ./ 
                        path_norm = path.lstrip("./")
                        desc_by_path[path_norm] = enriched

                    if qid and doc_name:
                        # e.g. ("COSTCO_2021_10K", "image7")
                        num = re.search(r"\d+", qid)
                        if num:
                            desc_by_docimg[(doc_name, num.group())] = enriched
                        desc_by_qid[f"{doc_name}_{qid}"] = enriched

                records_read += 1

        log.info("  Running total: %d descriptions", len(desc_by_path))

    log.info("Built description map: %d by path, %d by doc+img, from %d files",
             len(desc_by_path), len(desc_by_docimg), files_read)

    return desc_by_path, desc_by_docimg, desc_by_qid


# ------------------------------------------------------------------ #
# Step 2: Read all image points from Qdrant
# ------------------------------------------------------------------ #

def fetch_all_image_points(client: QdrantClient) -> list[dict]:
    """Scroll through all points in image_collection."""
    points    = []
    offset    = None
    page_size = 100

    while True:
        results, next_offset = client.scroll(
            collection_name=config.IMAGE_COLLECTION,
            limit=page_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points.extend(results)
        if next_offset is None or len(results) == 0:
            break
        offset = next_offset
        if len(points) % 500 == 0:
            log.info("  Fetched %d points so far...", len(points))

    log.info("Total image points in Qdrant: %d", len(points))
    return points


# ------------------------------------------------------------------ #
# Step 3: Match each Qdrant point to a description
# ------------------------------------------------------------------ #

def match_description(
    point_payload: dict,
    desc_by_path:  dict,
    desc_by_docimg: dict,
    desc_by_qid:   dict,
) -> str | None:
    """
    Try multiple matching strategies to find the right description.
    Returns the best description found, or None.
    """
    doc_name   = point_payload.get("doc_name", "")
    image_path = point_payload.get("image_path", "")
    quote_id   = point_payload.get("quote_id", "")

    # Strategy 1: exact image_path match
    if image_path:
        path_norm = image_path.lstrip("./")
        if path_norm in desc_by_path:
            return desc_by_path[path_norm]
        # Try without "images/" prefix
        basename = Path(image_path).name
        for p, d in desc_by_path.items():
            if Path(p).name == basename:
                return d

    # Strategy 2: doc_name + image number from quote_id
    if doc_name and quote_id:
        num = re.search(r"\d+", quote_id)
        if num:
            key = (doc_name, num.group())
            if key in desc_by_docimg:
                return desc_by_docimg[key]

    # Strategy 3: doc_name + image number from image_path filename
    if doc_name and image_path:
        num = re.search(r"image(\d+)", image_path)
        if num:
            key = (doc_name, num.group(1))
            if key in desc_by_docimg:
                return desc_by_docimg[key]

    # Strategy 4: composite quote_id key
    if doc_name and quote_id:
        key = f"{doc_name}_{quote_id}"
        if key in desc_by_qid:
            return desc_by_qid[key]

    return None


# ------------------------------------------------------------------ #
# Step 4: Bulk update Qdrant
# ------------------------------------------------------------------ #

def bulk_update(
    client:        QdrantClient,
    points:        list,
    desc_by_path:  dict,
    desc_by_docimg: dict,
    desc_by_qid:   dict,
    batch_size:    int  = 50,
    dry_run:       bool = False,
) -> dict:
    """Update all matchable points. Returns stats."""
    matched   = 0
    unmatched = 0
    skipped   = 0   # already has a good description
    updated   = 0

    update_batch = []   # list of (point_id, new_description)

    for p in points:
        payload = p.payload or {}
        pid     = p.id

        desc = match_description(payload, desc_by_path, desc_by_docimg, desc_by_qid)

        if desc is None:
            unmatched += 1
            continue

        # Skip if already has this exact description
        current = payload.get("text", "")
        if current and len(current) > 50 and current == desc:
            skipped += 1
            continue

        matched += 1
        update_batch.append((pid, desc))

        # Flush batch
        if len(update_batch) >= batch_size:
            if not dry_run:
                for point_id, new_desc in update_batch:
                    client.set_payload(
                        collection_name=config.IMAGE_COLLECTION,
                        payload={"text": new_desc},
                        points=[point_id],
                    )
            updated += len(update_batch)
            log.info("  Updated %d points (total so far: %d)", len(update_batch), updated)
            update_batch = []

    # Final flush
    if update_batch:
        if not dry_run:
            for point_id, new_desc in update_batch:
                client.set_payload(
                    collection_name=config.IMAGE_COLLECTION,
                    payload={"text": new_desc},
                    points=[point_id],
                )
        updated += len(update_batch)

    return {
        "total_points": len(points),
        "matched":      matched,
        "updated":      updated,
        "unmatched":    unmatched,
        "skipped":      skipped,
        "dry_run":      dry_run,
    }


# ------------------------------------------------------------------ #
# Also update text_collection with better chunk text
# ------------------------------------------------------------------ #

def build_text_desc_map() -> dict[str, str]:
    """
    Build map: (doc_name, quote_id) → text content
    from the flat-format source files.
    Used to update text_collection payloads too.
    """
    text_map = {}
    for fname in SOURCE_FILES:
        fpath = RAW_DIR / fname
        if not fpath.exists():
            continue
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except:
                    continue
                doc_name = rec.get("doc_name", "")
                for tq in rec.get("text_quotes", []):
                    qid  = tq.get("quote_id", "")
                    text = tq.get("text", "").strip()
                    if doc_name and qid and text:
                        text_map[(doc_name, qid)] = text
    log.info("Text description map: %d entries", len(text_map))
    return text_map


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run",    action="store_true",
                        help="Preview only — do not write to Qdrant")
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--images_only", action="store_true",
                        help="Only update image_collection (skip text_collection)")
    args = parser.parse_args()

    if args.dry_run:
        log.info("DRY RUN MODE — no changes will be written to Qdrant")

    # Connect
    client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
    log.info("Connected to Qdrant at %s:%s", config.QDRANT_HOST, config.QDRANT_PORT)

    # Build description maps from source files
    log.info("Building description maps from source files...")
    desc_by_path, desc_by_docimg, desc_by_qid = build_description_map()

    if not desc_by_path and not desc_by_docimg:
        log.error("No descriptions found in source files.")
        log.error("Make sure dev_20.jsonl / evaluation_20.jsonl exist in data/raw/")
        return

    # Fetch all image points from Qdrant
    log.info("Fetching all image points from Qdrant...")
    image_points = fetch_all_image_points(client)

    if not image_points:
        log.error("No image points found in Qdrant image_collection")
        return

    # Bulk update image_collection
    log.info("Matching and updating image descriptions...")
    stats = bulk_update(
        client, image_points,
        desc_by_path, desc_by_docimg, desc_by_qid,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )

    # Print summary
    log.info("\n" + "="*50)
    log.info("BULK UPDATE COMPLETE")
    log.info("="*50)
    log.info("Total image points:  %d", stats["total_points"])
    log.info("Matched + updated:   %d (%.1f%%)",
             stats["updated"],
             100 * stats["updated"] / max(stats["total_points"], 1))
    log.info("Unmatched (no desc): %d", stats["unmatched"])
    log.info("Skipped (unchanged): %d", stats["skipped"])
    if args.dry_run:
        log.info("DRY RUN — run without --dry_run to apply changes")
    else:
        log.info("All descriptions updated in Qdrant image_collection")
        log.info("BM25 will now find images by their visual content descriptions")
        log.info("\nTest it: python scripts/rag_pipeline.py --arch 6 --mode multimodal "
                 "--query \"national gas price peak summer 2022\"")


if __name__ == "__main__":
    main()