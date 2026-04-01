"""
debug_upload.py - Test document upload directly
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from document_ingestor import ingest_pdf
except ImportError:
    print("❌ Could not import document_ingestor")
    sys.exit(1)

import config

def test_upload():
    print("=" * 60)
    print("🔍 Testing Document Upload")
    print("=" * 60)
    
    # Check if sample_docs exists
    sample_docs = Path(config.DOCUMENT_FOLDER)
    print(f"\n📁 Sample docs folder: {sample_docs}")
    print(f"   Exists: {sample_docs.exists()}")
    
    if sample_docs.exists():
        # List PDF files
        pdf_files = list(sample_docs.glob("*.pdf"))
        print(f"\n📄 PDF files found: {len(pdf_files)}")
        for pdf in pdf_files:
            print(f"   - {pdf.name}")
        
        if pdf_files:
            # Try to upload the first PDF
            test_pdf = pdf_files[0]
            print(f"\n📤 Testing upload of: {test_pdf.name}")
            
            success, message = ingest_pdf(str(test_pdf), "TEST_DOCUMENT")
            print(f"\n✅ Success: {success}")
            print(f"📝 Message: {message}")
        else:
            print("\n❌ No PDF files found in sample_docs")
            print("   Please add a PDF file to:", sample_docs)
    else:
        print(f"\n❌ Creating sample_docs folder...")
        sample_docs.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {sample_docs}")
        print("   Please add a PDF file and run again")

if __name__ == "__main__":
    test_upload()