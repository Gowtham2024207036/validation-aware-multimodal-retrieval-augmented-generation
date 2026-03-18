"""
update_image_description.py
----------------------------
Update the text description stored in Qdrant for an image.
This improves BM25 retrieval so natural language queries find the image
without needing to mention the exact doc_name.

Usage:
    python scripts/update_image_description.py \
        --doc_name "2309.17421v2" \
        --image_path "images/2309.17421v2_image39.jpg" \
        --description "National gas price comparison chart 2019 to 2023. \
Line graph showing average US gas prices per gallon by year. \
The 2022 line peaks near $5.00 per gallon during summer June July 2022. \
2019 prices around $2.50. 2020 prices dipped to $1.75 during COVID. \
2021 prices rose back to $3.00. 2023 prices at $3.32 as of January 2023. \
Source: AAA GasPrices."

    # Or update all images matching a doc_name:
    python scripts/update_image_description.py --doc_name "2309.17421v2" --list
"""

import os
import sys
import argparse
import logging
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def list_images_for_doc(client: QdrantClient, doc_name: str):
    """List all image chunks stored for a given doc_name."""
    results, _ = client.scroll(
        collection_name=config.IMAGE_COLLECTION,
        scroll_filter=qmodels.Filter(
            must=[qmodels.FieldCondition(
                key="doc_name",
                match=qmodels.MatchValue(value=doc_name)
            )]
        ),
        limit=50,
        with_payload=True,
        with_vectors=False,
    )
    print(f"\nImages in Qdrant for doc_name='{doc_name}':")
    print(f"{'ID':<10} {'quote_id':<35} {'image_path':<50}")
    print("-" * 95)
    for p in results:
        print(f"{p.id:<10} {p.payload.get('quote_id',''):<35} "
              f"{p.payload.get('image_path',''):<50}")
    print(f"\nTotal: {len(results)} images")
    return results


def update_description(
    client:      QdrantClient,
    doc_name:    str,
    image_path:  str,
    description: str,
):
    """
    Update the 'text' field in Qdrant payload for a specific image.
    This improves BM25 retrieval — the description becomes searchable text.
    """
    # Find the point
    results, _ = client.scroll(
        collection_name=config.IMAGE_COLLECTION,
        scroll_filter=qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="doc_name",
                    match=qmodels.MatchValue(value=doc_name)
                ),
                qmodels.FieldCondition(
                    key="image_path",
                    match=qmodels.MatchValue(value=image_path)
                ),
            ]
        ),
        limit=5,
        with_payload=True,
        with_vectors=False,
    )

    if not results:
        log.error("No image found for doc_name='%s' image_path='%s'", doc_name, image_path)
        log.error("Run with --list to see available images.")
        return False

    # Also update in text_collection if a text chunk exists for this doc
    for point in results:
        client.set_payload(
            collection_name=config.IMAGE_COLLECTION,
            payload={"text": description},
            points=[point.id],
        )
        log.info("Updated IMAGE collection point %d", point.id)

    # Update or create a text chunk for BM25 searchability
    # Search text collection for this doc
    text_results, _ = client.scroll(
        collection_name=config.TEXT_COLLECTION,
        scroll_filter=qmodels.Filter(
            must=[qmodels.FieldCondition(
                key="doc_name",
                match=qmodels.MatchValue(value=doc_name)
            )]
        ),
        limit=5,
        with_payload=True,
        with_vectors=False,
    )

    if text_results:
        # Update existing text chunk
        client.set_payload(
            collection_name=config.TEXT_COLLECTION,
            payload={"text": description},
            points=[text_results[0].id],
        )
        log.info("Updated TEXT collection point %d with new description", text_results[0].id)
    else:
        log.info("No text chunk found for this doc — BM25 won't find it by content.")
        log.info("Re-ingest with ingest_document.py to create a text chunk.")

    log.info("Description updated successfully.")
    log.info("Now query: 'what was the gas price peak in summer 2022?' — BM25 will find it.")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc_name",    required=True, help="doc_name in Qdrant")
    parser.add_argument("--image_path",  default=None,  help="image_path in Qdrant payload")
    parser.add_argument("--description", default=None,  help="New text description")
    parser.add_argument("--list",        action="store_true",
                        help="List all images for this doc_name")
    args = parser.parse_args()

    client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
    log.info("Connected to Qdrant at %s:%s", config.QDRANT_HOST, config.QDRANT_PORT)

    if args.list or not args.image_path:
        list_images_for_doc(client, args.doc_name)
        return

    if not args.description:
        print("Please provide --description")
        return

    update_description(client, args.doc_name, args.image_path, args.description)


if __name__ == "__main__":
    main()