import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieve import Retriever


def main():
    # Initialize retriever
    retriever = Retriever()
    
    # Get user question
    question = input("\nAsk a question: ")
    
    # Search text collection
    print("\nSearching text collection...")
    text_results = retriever.search_text(question)
    
    print(f"\nFound {len(text_results)} text results:")
    print("-" * 60)
    for i, r in enumerate(text_results, 1):
        print(f"\nResult {i}:")
        print(f"  Score: {r.score:.4f}")
        print(f"  Payload: {r.payload}")
    
    # Search image collection
    print("\n" + "="*60)
    print("Searching image collection...")
    image_results = retriever.search_images_by_text(question)  # Fixed method name
    
    print(f"\nFound {len(image_results)} image results:")
    print("-" * 60)
    for i, r in enumerate(image_results, 1):
        print(f"\nImage {i}:")
        print(f"  Score: {r.score:.4f}")
        print(f"  Payload: {r.payload}")


if __name__ == "__main__":
    main()