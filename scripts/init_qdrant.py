# scripts/init_qdrant.py
"""
Initialize Qdrant collections for the multimodal RAG system.

Run this script once before ingesting any embeddings:
    python scripts/init_qdrant.py
"""

import sys
from pathlib import Path

# Allow imports from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_manager import QdrantManager


def main():
    manager = QdrantManager()
    manager.create_collections()


if __name__ == "__main__":
    main()
