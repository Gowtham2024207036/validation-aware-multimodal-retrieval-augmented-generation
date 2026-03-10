import fitz
import pdfplumber
from PIL import Image
import io
import hashlib
from typing import List, Dict, Any


class DocumentParser:
    """
    Enhanced PDF document parser for extracting text, images, and tables.
    Includes error handling, image deduplication, and better structure.
    """
    
    def __init__(self, pdf_path: str):
        """
        Initialize the document parser.
        
        Args:
            pdf_path: Path to the PDF file
        """
        self.pdf_path = pdf_path
        self._validate_pdf()
    
    def _validate_pdf(self):
        """Validate that the PDF file exists and can be opened."""
        try:
            doc = fitz.open(self.pdf_path)
            doc.close()
        except Exception as e:
            raise ValueError(f"Invalid PDF file: {self.pdf_path}. Error: {e}")
    
    def extract_text(self) -> List[Dict[str, Any]]:
        """
        Extract text blocks from PDF pages.
        
        Returns:
            List of dicts with keys: page, content, char_count
        """
        text_blocks = []
        doc = None
        
        try:
            doc = fitz.open(self.pdf_path)
            
            for page_num, page in enumerate(doc):
                text = page.get_text("text")  # Specify extraction method
                
                if text.strip():
                    text_blocks.append({
                        "page": page_num + 1,  # 1-indexed for user-friendliness
                        "content": text.strip(),
                        "char_count": len(text.strip())
                    })
            
            print(f"✓ Extracted text from {len(text_blocks)} pages")
            
        except Exception as e:
            print(f"✗ Error extracting text: {e}")
            raise
        finally:
            if doc:
                doc.close()
        
        return text_blocks
    
    def extract_images(self, min_width: int = 100, min_height: int = 100) -> List[Dict[str, Any]]:
        """
        Extract images from PDF with deduplication.
        
        Args:
            min_width: Minimum image width to extract (filters out tiny images/icons)
            min_height: Minimum image height to extract
            
        Returns:
            List of dicts with keys: page, image, width, height, hash
        """
        images = []
        seen_hashes = set()
        doc = None
        total_images_processed = 0
        
        try:
            doc = fitz.open(self.pdf_path)
            
            for page_index in range(len(doc)):
                page = doc[page_index]
                image_list = page.get_images(full=True)
                
                for img_index, img in enumerate(image_list):
                    total_images_processed += 1
                    try:
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        
                        # Open image and get dimensions
                        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                        width, height = image.size
                        
                        # Filter out small images (icons, decorations)
                        if width < min_width or height < min_height:
                            continue
                        
                        # Deduplicate images using hash
                        img_hash = hashlib.md5(image_bytes).hexdigest()
                        if img_hash in seen_hashes:
                            continue
                        
                        seen_hashes.add(img_hash)
                        images.append({
                            "page": page_index + 1,  # 1-indexed
                            "image": image,
                            "width": width,
                            "height": height,
                            "hash": img_hash,
                            "format": base_image.get("ext", "unknown")
                        })
                        
                    except Exception as e:
                        print(f"Warning: Skipping image on page {page_index + 1}: {e}")
                        continue
            
            filtered_count = total_images_processed - len(images)
            print(f"✓ Extracted {len(images)} unique images (filtered {filtered_count} duplicates/small images)")
            
        except Exception as e:
            print(f"✗ Error extracting images: {e}")
            raise
        finally:
            if doc:
                doc.close()
        
        return images
    
    def extract_tables(self, min_rows: int = 2) -> List[Dict[str, Any]]:
        """
        Extract tables from PDF pages.
        
        Args:
            min_rows: Minimum number of rows to consider a valid table
            
        Returns:
            List of dicts with keys: page, headers, rows, row_count, col_count
        """
        tables_data = []
        
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                for page_index, page in enumerate(pdf.pages):
                    tables = page.extract_tables()
                    
                    for table_index, table in enumerate(tables):
                        if not table or len(table) < min_rows:
                            continue
                        
                        # Handle missing or malformed headers
                        headers = table[0] if table[0] else [f"Column_{i}" for i in range(len(table[0]))]
                        
                        # Clean headers (remove None, empty strings)
                        headers = [h if h else f"Column_{i}" for i, h in enumerate(headers)]
                        
                        # Process rows
                        rows = []
                        for row_data in table[1:]:
                            # Skip empty rows
                            if not any(row_data):
                                continue
                            
                            # Ensure row has same length as headers
                            row_data = list(row_data)
                            while len(row_data) < len(headers):
                                row_data.append(None)
                            
                            row_dict = dict(zip(headers, row_data[:len(headers)]))
                            rows.append(row_dict)
                        
                        if rows:  # Only add if we have valid rows
                            tables_data.append({
                                "page": page_index + 1,  # 1-indexed
                                "table_index": table_index,
                                "headers": headers,
                                "rows": rows,
                                "row_count": len(rows),
                                "col_count": len(headers)
                            })
            
            print(f"✓ Extracted {len(tables_data)} tables")
            
        except Exception as e:
            print(f"✗ Error extracting tables: {e}")
            raise
        
        return tables_data
    
    def extract_all(self, min_img_width: int = 100, min_img_height: int = 100, 
                    min_table_rows: int = 2) -> Dict[str, Any]:
        """
        Extract all content (text, images, tables) from the PDF.
        
        Args:
            min_img_width: Minimum image width
            min_img_height: Minimum image height
            min_table_rows: Minimum table rows
            
        Returns:
            Dict with keys: text_blocks, images, tables, metadata
        """
        print(f"\n{'='*60}")
        print(f"Parsing PDF: {self.pdf_path}")
        print(f"{'='*60}")
        
        text_blocks = self.extract_text()
        images = self.extract_images(min_img_width, min_img_height)
        tables = self.extract_tables(min_table_rows)
        
        # Metadata
        doc = None
        try:
            doc = fitz.open(self.pdf_path)
            metadata = {
                "page_count": len(doc),
                "title": doc.metadata.get("title", "Unknown"),
                "author": doc.metadata.get("author", "Unknown")
            }
        except:
            metadata = {"page_count": 0}
        finally:
            if doc:
                doc.close()
        
        print(f"\n{'='*60}")
        print(f"Extraction Summary:")
        print(f"  • Text blocks: {len(text_blocks)}")
        print(f"  • Images: {len(images)}")
        print(f"  • Tables: {len(tables)}")
        print(f"  • Pages: {metadata.get('page_count', 0)}")
        print(f"{'='*60}\n")
        
        return {
            "text_blocks": text_blocks,
            "images": images,
            "tables": tables,
            "metadata": metadata
        }


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python document_parser.py <path_to_pdf>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    parser = DocumentParser(pdf_path)
    
    # Extract everything
    results = parser.extract_all()
    
    # Display sample results
    if results["text_blocks"]:
        print("\nSample text (first 200 chars):")
        print(results["text_blocks"][0]["content"][:200] + "...")
    
    if results["images"]:
        print(f"\nFirst image: {results['images'][0]['width']}x{results['images'][0]['height']}")
    
    if results["tables"]:
        print(f"\nFirst table: {results['tables'][0]['row_count']} rows × {results['tables'][0]['col_count']} cols")