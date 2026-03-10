import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from document_parser import DocumentParser
from embedding_utils import embed_texts, embed_images
from qdrant_utils import QdrantManager


def index_document(pdf_path):
    """
    Index a PDF document by extracting and embedding its content.
    
    Args:
        pdf_path: Path to the PDF file
    """
    print("\n" + "="*60)
    print("INDEXING DOCUMENT")
    print("="*60)
    print(f"File: {os.path.basename(pdf_path)}\n")
    
    # Validate file exists
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    try:
        # Parse document
        parser = DocumentParser(pdf_path)
        data = parser.extract_all()
        
        # Fix: Use correct keys from parser output
        text_blocks = data["text_blocks"]  # NOT "texts"
        images = data["images"]
        tables = data["tables"]  # Also available if needed
        
        if not text_blocks and not images:
            print("⚠ Warning: No content extracted from PDF")
            return
        
        # Initialize Qdrant manager
        qdrant = QdrantManager()
        
        # Process text blocks
        if text_blocks:
            print(f"\n📝 Processing {len(text_blocks)} text blocks...")
            text_contents = [t["content"] for t in text_blocks]
            text_vectors = embed_texts(text_contents, batch_size=8)
            
            # Create payloads with metadata
            text_payloads = []
            for i, t in enumerate(text_blocks):
                text_payloads.append({
                    "type": "text",
                    "content": t["content"][:500],  # Store first 500 chars for preview
                    "page": t["page"],
                    "char_count": t["char_count"],
                    "document": os.path.basename(pdf_path),
                    "document_path": pdf_path
                })
            
            # Insert into Qdrant
            qdrant.insert_vectors(
                config.TEXT_COLLECTION,
                text_vectors,
                text_payloads
            )
            print(f"✓ Indexed {len(text_blocks)} text blocks")
        else:
            print("⚠ No text blocks to index")
        
        # Process images
        if images:
            print(f"\n🖼️  Processing {len(images)} images...")
            image_objects = [img["image"] for img in images]
            image_vectors = embed_images(image_objects, batch_size=2)
            
            # Create payloads with metadata
            image_payloads = []
            for i, img in enumerate(images):
                image_payloads.append({
                    "type": "image",
                    "page": img["page"],
                    "width": img["width"],
                    "height": img["height"],
                    "format": img["format"],
                    "hash": img["hash"],
                    "document": os.path.basename(pdf_path),
                    "document_path": pdf_path
                })
            
            # Insert into Qdrant
            qdrant.insert_vectors(
                config.IMAGE_COLLECTION,
                image_vectors,
                image_payloads
            )
            print(f"✓ Indexed {len(images)} images")
        else:
            print("⚠ No images to index")
        
        # Process tables (optional)
        if tables:
            print(f"\n📊 Found {len(tables)} tables (not indexed in this version)")
            # You can add table indexing here if needed
        
        print("\n" + "="*60)
        print("INDEXING COMPLETE ✓")
        print("="*60)
        print(f"Summary:")
        print(f"  • Text blocks indexed: {len(text_blocks)}")
        print(f"  • Images indexed: {len(images)}")
        print(f"  • Tables found: {len(tables)}")
        
    except Exception as e:
        print(f"\n❌ Error indexing document: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    # Test with a single PDF
    import glob
    
    pdf_files = glob.glob(os.path.join(config.DOCUMENT_FOLDER, "*.pdf"))
    if not pdf_files:
        print("No PDF files found in:", config.DOCUMENT_FOLDER)
    else:
        print(f"Found {len(pdf_files)} PDF(s)")
        index_document(pdf_files[0])