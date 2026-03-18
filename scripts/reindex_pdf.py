"""
reindex_pdf.py
--------------
Deletes existing chunks for a document and re-indexes from scratch.
Use this when a PDF was indexed with bad text extraction (e.g. Tamil garbled).

Usage:
    python scripts/reindex_pdf.py --pdf data/raw/TNEA_brochure.pdf --doc_name TNEA_2025
    python scripts/reindex_pdf.py --list   # show all indexed doc names
    python scripts/reindex_pdf.py --delete --doc_name 2_INFORMATION_BROCHURE  # delete only
"""

import os, sys, argparse, logging
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def list_docs(client: QdrantClient):
    """List all doc_names in both collections."""
    from collections import defaultdict
    counts = defaultdict(lambda: {"text": 0, "image": 0})

    for col, modality in [(config.TEXT_COLLECTION, "text"),
                          (config.IMAGE_COLLECTION, "image")]:
        offset = None
        while True:
            results, next_offset = client.scroll(
                collection_name=col, limit=200, offset=offset,
                with_payload=True, with_vectors=False)
            for p in results:
                dn = p.payload.get("doc_name", "")
                if dn:
                    counts[dn][modality] += 1
            if next_offset is None or not results:
                break
            offset = next_offset

    print(f"\n{'Doc Name':<50} {'Text':>6} {'Image':>6}")
    print("-" * 65)
    for dn in sorted(counts.keys()):
        print(f"{dn:<50} {counts[dn]['text']:>6} {counts[dn]['image']:>6}")
    print(f"\nTotal: {len(counts)} documents")


def delete_doc(client: QdrantClient, doc_name: str):
    """Delete all chunks for a given doc_name from both collections."""
    for col in [config.TEXT_COLLECTION, config.IMAGE_COLLECTION]:
        client.delete(
            collection_name=col,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(must=[
                    qmodels.FieldCondition(
                        key="doc_name",
                        match=qmodels.MatchValue(value=doc_name)
                    )
                ])
            )
        )
        log.info("Deleted chunks for '%s' from %s", doc_name, col)


def reindex(pdf_path: str, doc_name: str, client: QdrantClient):
    """Delete old chunks and re-index with pymupdf."""
    # Step 1: Delete old chunks
    log.info("Deleting old chunks for '%s'...", doc_name)
    delete_doc(client, doc_name)

    # Step 2: Re-index
    log.info("Re-indexing '%s' from %s...", doc_name, pdf_path)
    from ingest_document import ingest
    ingest(pdf_path, doc_name)
    log.info("Done. '%s' re-indexed successfully.", doc_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list",     action="store_true", help="List all indexed documents")
    parser.add_argument("--delete",   action="store_true", help="Delete doc_name only, no re-index")
    parser.add_argument("--pdf",      default=None,        help="Path to PDF file")
    parser.add_argument("--doc_name", default=None,        help="Document name in Qdrant")
    args = parser.parse_args()

    client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
    log.info("Connected to Qdrant")

    if args.list:
        list_docs(client)
        return

    if not args.doc_name:
        print("Please provide --doc_name")
        return

    if args.delete:
        delete_doc(client, args.doc_name)
        log.info("Deleted. Run with --list to verify.")
        return

    if not args.pdf:
        print("Please provide --pdf path")
        return

    reindex(args.pdf, args.doc_name, client)


if __name__ == "__main__":
    main()