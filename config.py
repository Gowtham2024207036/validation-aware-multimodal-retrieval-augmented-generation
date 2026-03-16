import torch
import os

# -----------------------------
# Base Directory (where config.py is located)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------
# Qdrant Configuration
# -----------------------------
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333


TEXT_COLLECTION = "text_collection"
IMAGE_COLLECTION = "image_collection"
TABLE_COLLECTION = "table_collection"

DATA_DIR = "data/raw"
IMAGE_DIR = os.path.join(DATA_DIR, "images")
# -----------------------------
# Model Configuration
# -----------------------------
TEXT_MODEL = "sentence-transformers/all-mpnet-base-v2"
IMAGE_MODEL = "openai/clip-vit-base-patch32"

# -----------------------------
# Vector Dimensions
# -----------------------------
TEXT_VECTOR_SIZE = 768
IMAGE_VECTOR_SIZE = 512

# -----------------------------
# Retrieval Configuration
# -----------------------------
TOP_K_TEXT = 8
TOP_K_IMAGE = 4
TOP_K_TABLE = 3
RERANK_TOP_K = 5

# -----------------------------
# Device Configuration
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Paths (Absolute paths based on config.py location)
# -----------------------------
DOCUMENT_FOLDER = os.path.join(BASE_DIR, "sample_docs")
VECTOR_STORAGE = os.path.join(BASE_DIR, "storage")