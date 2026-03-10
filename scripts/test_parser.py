import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from document_parser import DocumentParser


def display_text_sample(texts, max_length=300):
    """Display a sample of extracted text."""
    if texts:
        print("\n" + "="*60)
        print("TEXT SAMPLE (First block):")
        print("="*60)
        sample = texts[0]["content"][:max_length]
        print(f"Page {texts[0]['page']}: {sample}...")
        print(f"Total characters: {texts[0]['char_count']}")
    else:
        print("\n⚠ No text blocks extracted")


def display_image_info(images):
    """Display information about extracted images."""
    if images:
        print("\n" + "="*60)
        print("IMAGE DETAILS:")
        print("="*60)
        for i, img_data in enumerate(images[:3], 1):  # Show first 3
            print(f"Image {i}:")
            print(f"  • Page: {img_data['page']}")
            print(f"  • Size: {img_data['width']}x{img_data['height']}")
            print(f"  • Format: {img_data['format']}")
            print(f"  • Hash: {img_data['hash'][:16]}...")
        if len(images) > 3:
            print(f"  ... and {len(images) - 3} more images")
    else:
        print("\n⚠ No images extracted")


def display_table_info(tables):
    """Display information about extracted tables."""
    if tables:
        print("\n" + "="*60)
        print("TABLE DETAILS:")
        print("="*60)
        for i, table in enumerate(tables, 1):
            print(f"Table {i}:")
            print(f"  • Page: {table['page']}")
            print(f"  • Dimensions: {table['row_count']} rows × {table['col_count']} columns")
            print(f"  • Headers: {', '.join(table['headers'][:5])}")
            if table['col_count'] > 5:
                print(f"    ... and {table['col_count'] - 5} more columns")
            
            # Show first row as sample
            if table['rows']:
                print(f"  • Sample row 1:")
                for key, value in list(table['rows'][0].items())[:3]:
                    print(f"      {key}: {value}")
    else:
        print("\n⚠ No tables extracted")


def main():
    print("\n" + "="*60)
    print("PDF PARSER TEST")
    print("="*60)
    
    # Find PDF files
    pdf_files = []
    for f in os.listdir(config.DOCUMENT_FOLDER):
        if f.lower().endswith(".pdf"):
            pdf_files.append(os.path.join(config.DOCUMENT_FOLDER, f))

    if len(pdf_files) == 0:
        print(f"\n❌ No PDF files found in: {config.DOCUMENT_FOLDER}")
        print("Please add a PDF file to the sample_docs folder and try again.")
        return

    # Test the first PDF
    pdf_path = pdf_files[0]
    pdf_name = os.path.basename(pdf_path)
    
    print(f"\n📄 Testing with: {pdf_name}")
    print(f"   Path: {pdf_path}")
    
    try:
        # Parse the PDF
        parser = DocumentParser(pdf_path)
        
        # Extract all content
        print("\n" + "-"*60)
        print("EXTRACTING CONTENT...")
        print("-"*60)
        
        texts = parser.extract_text()
        images = parser.extract_images(min_width=100, min_height=100)
        tables = parser.extract_tables(min_rows=2)
        
        # Summary
        print("\n" + "="*60)
        print("EXTRACTION SUMMARY:")
        print("="*60)
        print(f"✓ Text blocks: {len(texts)}")
        print(f"✓ Images: {len(images)}")
        print(f"✓ Tables: {len(tables)}")
        
        # Detailed output
        display_text_sample(texts)
        display_image_info(images)
        display_table_info(tables)
        
        # Alternative: Use extract_all() method
        print("\n" + "="*60)
        print("TESTING extract_all() METHOD:")
        print("="*60)
        
        all_content = parser.extract_all()
        print(f"\n✓ Successfully extracted all content")
        print(f"  • Metadata: {all_content['metadata']}")
        
    except Exception as e:
        print(f"\n❌ Error during parsing: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*60)
    print("TEST COMPLETE ✓")
    print("="*60)


if __name__ == "__main__":
    main()