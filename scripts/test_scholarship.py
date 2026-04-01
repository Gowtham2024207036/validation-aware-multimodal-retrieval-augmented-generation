"""
test_scholarship.py - Test if scholarship information is being retrieved correctly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_retriever import SharedModels
import arch6_full_proposed as best_arch

def test_scholarship_query():
    print("=" * 60)
    print("🔍 Testing Scholarship Query")
    print("=" * 60)
    
    query = "What scholarships are available for SC ST students in TNEA?"
    
    models = SharedModels()
    result = best_arch.retrieve(query, models)
    
    text_hits = result.get("text", [])
    image_hits = result.get("image", [])
    
    print(f"\n📚 Retrieved: {len(text_hits)} text chunks, {len(image_hits)} images")
    
    # Look for specific scholarship sections
    scholarship_keywords = [
        "tuition fee concession",
        "first graduate",
        "post matric",
        "scholarship",
        "fee waiver",
        "AICTE",
        "TFW"
    ]
    
    print("\n📄 Text chunks containing scholarship information:")
    found = False
    for i, h in enumerate(text_hits, 1):
        text = h.get("text", "").lower()
        doc = h.get("doc_name", "Unknown")
        page = h.get("page_id", "?")
        
        for keyword in scholarship_keywords:
            if keyword in text:
                found = True
                print(f"\n{i}. [{doc}, Page {page}]")
                # Show the relevant snippet
                snippet = h.get("text", "")[:500]
                print(f"   {snippet}...")
                break
    
    if not found:
        print("❌ No scholarship information found in retrieved chunks!")
    
    # Check images
    print("\n🖼️ Images retrieved:")
    for i, h in enumerate(image_hits, 1):
        doc = h.get("doc_name", "Unknown")
        page = h.get("page_id", "?")
        print(f"{i}. [{doc}, Page {page}]")
    
    return text_hits, image_hits

if __name__ == "__main__":
    test_scholarship_query()