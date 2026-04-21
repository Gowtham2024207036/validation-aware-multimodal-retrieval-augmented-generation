import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from qdrant_client import QdrantClient

client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)

# List all images (limit 20)
results = client.scroll(
    collection_name=config.IMAGE_COLLECTION,
    limit=20,
    with_payload=True
)

for point in results[0]:
    print(f"ID: {point.id}, doc_name: {point.payload.get('doc_name')}, path: {point.payload.get('image_path')}")