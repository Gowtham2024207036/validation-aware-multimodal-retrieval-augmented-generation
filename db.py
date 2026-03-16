from qdrant_client import QdrantClient

print("🧹 Sweeping old data out of Qdrant...")
client = QdrantClient("http://localhost:6333")

# Delete the contaminated collections
collections_to_delete = ["text_collection", "image_collection", "table_collection"]

for col in collections_to_delete:
    try:
        client.delete_collection(col)
        print(f"Deleted: {col}")
    except Exception as e:
        print(f"Skipped {col} (didn't exist)")

print("✅ Qdrant is completely clean and ready for fresh indexing!")