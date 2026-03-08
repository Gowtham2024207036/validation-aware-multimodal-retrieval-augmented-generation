# Qdrant connection settings
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

# Collection names
TEXT_COLLECTION = "text_collection"
IMAGE_COLLECTION = "image_collection"
TABLE_COLLECTION = "table_collection"

# Vector sizes
# all-mpnet-base-v2 produces 768-dimensional embeddings
TEXT_VECTOR_SIZE = 768
# CLIP ViT-B/32 produces 512-dimensional embeddings
IMAGE_VECTOR_SIZE = 512
