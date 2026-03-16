import os
import sys
import logging
import torch
from transformers import CLIPProcessor, CLIPModel
from qdrant_client import QdrantClient

# Ensure Python can find the 'config' module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Suppress warnings for a clean output
logging.basicConfig(level=logging.ERROR)
import warnings
warnings.filterwarnings("ignore")

def test_cross_modal_retrieval():
    print("🚀 Initializing Cross-Modal Retriever Test...")
    
    # 1. Connect to Qdrant
    client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
    
    # 2. Load the CLIP Model
    print(f"🧠 Loading CLIP model for Text-to-Image matching on {config.DEVICE.upper()}...")
    model = CLIPModel.from_pretrained(config.IMAGE_MODEL).to(config.DEVICE)
    processor = CLIPProcessor.from_pretrained(config.IMAGE_MODEL)
    
    test_query = "what is Long-term Debt to Total Liabilities for COSTCO in FY2021?"
    print(f"\n🗣️  USER QUERY: '{test_query}'")
    
    # 3. Convert the text question into a 512-dimensional CLIP vector
    with torch.no_grad():
        inputs = processor(text=[test_query], return_tensors="pt", padding=True).to(config.DEVICE)
        text_features = model.get_text_features(**inputs)
        # Normalize the vector (essential for Cosine Distance)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        query_vector = text_features.cpu().numpy()[0].tolist()
    
    # 4. Search the Image Collection (Using the NEW Qdrant API)
    print("\n🔍 Searching the Image Vector Database (Comparing against 1,717 images)...")
    
    # FIXED: Qdrant v1.10+ removed .search() and replaced it with .query_points()
    response = client.query_points(
        collection_name=config.IMAGE_COLLECTION,
        query=query_vector,
        limit=3  # Bring back the top 3 best visual matches
    )
    
    print("\n✅ TOP 3 RETRIEVED IMAGES:")
    print("-" * 50)
    for i, hit in enumerate(response.points, 1):
        score = hit.score
        doc_name = hit.payload.get("doc_name")
        image_path = hit.payload.get("image_path")
        desc = hit.payload.get("text", "")[:100] + "..." # Truncate description
        
        print(f"Rank {i}: Score: {score:.4f} | Document: {doc_name}")
        print(f"Path  : {image_path}")
        print(f"Desc  : {desc}")
        print("-" * 50)

if __name__ == "__main__":
    test_cross_modal_retrieval()