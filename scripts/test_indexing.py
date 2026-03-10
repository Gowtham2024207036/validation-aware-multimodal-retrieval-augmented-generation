import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from index_document import index_document
from qdrant_utils import QdrantManager


def main():
    """
    Test indexing by processing all PDFs in the document folder.
    """
    print("\n" + "="*60)
    print("DOCUMENT INDEXING TEST")
    print("="*60)
    
    # Find all PDFs
    pdf_files = []
    for f in os.listdir(config.DOCUMENT_FOLDER):
        if f.lower().endswith(".pdf"):
            pdf_path = os.path.join(config.DOCUMENT_FOLDER, f)
            pdf_files.append(pdf_path)
    
    if not pdf_files:
        print(f"\n❌ No PDF files found in: {config.DOCUMENT_FOLDER}")
        print("Please add PDF files to the folder and try again.")
        return
    
    print(f"\n📁 Found {len(pdf_files)} PDF file(s):")
    for i, pdf in enumerate(pdf_files, 1):
        print(f"  {i}. {os.path.basename(pdf)}")
    
    # Initialize Qdrant (ensure collections exist)
    print("\n🔧 Initializing Qdrant collections...")
    try:
        qdrant = QdrantManager()
        print("✓ Qdrant ready")
    except Exception as e:
        print(f"❌ Error connecting to Qdrant: {e}")
        print("Make sure Qdrant is running on localhost:6333")
        return
    
    # Process each PDF
    success_count = 0
    fail_count = 0
    
    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"\n{'='*60}")
        print(f"Processing {i}/{len(pdf_files)}")
        print(f"{'='*60}")
        
        try:
            index_document(pdf_path)
            success_count += 1
        except Exception as e:
            print(f"❌ Failed to index: {os.path.basename(pdf_path)}")
            print(f"Error: {e}")
            fail_count += 1
            continue
    
    # Final summary
    print("\n" + "="*60)
    print("INDEXING SUMMARY")
    print("="*60)
    print(f"✓ Successfully indexed: {success_count} document(s)")
    if fail_count > 0:
        print(f"❌ Failed to index: {fail_count} document(s)")
    print("="*60)
    
    # Show collection stats
    try:
        print("\n📊 Collection Statistics:")
        text_info = qdrant.client.get_collection(config.TEXT_COLLECTION)
        image_info = qdrant.client.get_collection(config.IMAGE_COLLECTION)
        
        print(f"  • Text collection: {text_info.points_count} vectors")
        print(f"  • Image collection: {image_info.points_count} vectors")
    except Exception as e:
        print(f"⚠ Could not retrieve collection stats: {e}")


if __name__ == "__main__":
    main()