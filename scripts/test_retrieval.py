"""
test_retrieval.py - Test if documents are properly indexed
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_retriever import SharedModels
import config

def test_retrieval(query):
    print(f"\n{'='*60}")
    print(f"Testing query: '{query}'")
    print(f"{'='*60}")
    
    models = SharedModels()
    import arch6_full_proposed as best_arch
    
    result = best_arch.retrieve(query, models)
    text_hits = result.get("text", [])
    image_hits = result.get("image", [])
    
    print(f"\n📚 Retrieved: {len(text_hits)} text, {len(image_hits)} images")
    
    if text_hits:
        print("\n📄 Text chunks found:")
        for i, h in enumerate(text_hits[:3], 1):
            doc = h.get("doc_name", "Unknown")
            text = (h.get("text") or "")[:300]
            print(f"\n{i}. [{doc}]\n   {text}...")
    else:
        print("\n❌ No text chunks found!")
    
    if image_hits:
        print(f"\n🖼️ Images found: {len(image_hits)}")
        for i, h in enumerate(image_hits[:2], 1):
            doc = h.get("doc_name", "Unknown")
            path = h.get("image_path", "")
            print(f"   {i}. [{doc}] {path}")
    
    return text_hits, image_hits

def list_all_documents():
    """List all documents in the system"""
    print("\n" + "="*60)
    print("📚 All Indexed Documents")
    print("="*60)
    
    from qdrant_client import QdrantClient
    client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
    
    # Get all unique document names from text collection
    text_result, _ = client.scroll(
        collection_name=config.TEXT_COLLECTION,
        limit=1000,
        with_payload=True
    )
    
    docs = {}
    for point in text_result:
        doc_name = point.payload.get("doc_name", "Unknown")
        if doc_name not in docs:
            docs[doc_name] = {"text": 0, "image": 0}
        docs[doc_name]["text"] += 1
    
    # Get image counts
    image_result, _ = client.scroll(
        collection_name=config.IMAGE_COLLECTION,
        limit=1000,
        with_payload=True
    )
    
    for point in image_result:
        doc_name = point.payload.get("doc_name", "Unknown")
        if doc_name not in docs:
            docs[doc_name] = {"text": 0, "image": 0}
        docs[doc_name]["image"] += 1
    
    if docs:
        for doc_name, counts in docs.items():
            print(f"📄 {doc_name}: {counts['text']} text chunks, {counts['image']} images")
    else:
        print("❌ No documents found!")
    
    return docs

if __name__ == "__main__":
    # First, list all documents
    docs = list_all_documents()
    
    # Then test your query
    if "TNEA_2025" in str(docs) or "TNEA" in str(docs):
        print("\n✅ TNEA document found! Testing query...")
        test_retrieval("What scholarship is available for SC ST students?")
    else:
        print("\n❌ TNEA document not found. Please upload it first.")
        print("\nTo upload:")
        print("1. Run: python scripts/app.py")
        print("2. Go to the PDF tab")
        print("3. Upload your TNEA brochure PDF")
        print("4. Then run this test again")