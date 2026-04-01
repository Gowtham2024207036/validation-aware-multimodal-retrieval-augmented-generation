"""
prompt_builder.py - Unified prompt builder for hybrid RAG system
Supports both text-only and multimodal prompts
"""

# System prompt for the LLM
SYSTEM_PROMPT = (
    "You are a helpful assistant specialized in document understanding and question answering. "
    "Answer questions using ONLY the provided context from documents and images. "
    "If the answer is not in the context, say 'The answer is not found in the provided documents.' "
    "Always cite your sources by mentioning the document name and page number when available."
)


def build_text_only_prompt(query: str, text_hits: list, image_hits: list = None) -> tuple:
    """
    Build a text-only prompt from retrieved text chunks
    
    Args:
        query: User question
        text_hits: List of retrieved text chunks
        image_hits: List of retrieved image metadata (optional)
    
    Returns:
        tuple: (prompt_text, empty_list_for_compatibility)
    """
    context_parts = []
    
    # Add text context
    if text_hits:
        context_parts.append("=== RETRIEVED TEXT PASSAGES ===")
        for i, hit in enumerate(text_hits, 1):
            doc_name = hit.get('doc_name', 'Unknown Document').replace('_', ' ').title()
            page = hit.get('page_id', '')
            text = hit.get('text', '')[:1000]  # Limit text length
            
            page_str = f" [Page {page}]" if page else ""
            context_parts.append(f"\n[{i}] From: {doc_name}{page_str}\n{text}")
    
    # Add image descriptions if available
    if image_hits:
        context_parts.append("\n=== IMAGE DESCRIPTIONS ===")
        for i, hit in enumerate(image_hits, 1):
            doc_name = hit.get('doc_name', 'Unknown Document').replace('_', ' ').title()
            desc = hit.get('text', 'No description available')[:500]
            page = hit.get('page_id', '')
            
            page_str = f" [Page {page}]" if page else ""
            context_parts.append(f"\n[Image {i}] From: {doc_name}{page_str}\nDescription: {desc}")
    
    # Add question
    context_parts.append(f"\n=== QUESTION ===\n{query}")
    context_parts.append("\n=== ANSWER ===")
    
    return "\n\n".join(context_parts), []


def build_multimodal_prompt(query: str, text_hits: list, image_hits: list = None, max_images: int = 3) -> tuple:
    """
    Build a multimodal prompt with actual images
    
    Args:
        query: User question
        text_hits: List of retrieved text chunks
        image_hits: List of retrieved image metadata
        max_images: Maximum number of images to include
    
    Returns:
        tuple: (prompt_text, list_of_image_paths)
    """
    image_hits = image_hits or []
    context_parts = []
    image_paths = []
    
    # Add text context
    if text_hits:
        context_parts.append("=== RETRIEVED TEXT PASSAGES ===")
        for i, hit in enumerate(text_hits, 1):
            doc_name = hit.get('doc_name', 'Unknown Document').replace('_', ' ').title()
            page = hit.get('page_id', '')
            text = hit.get('text', '')[:800]  # Shorter for multimodal
            
            page_str = f" [Page {page}]" if page else ""
            context_parts.append(f"\n[{i}] From: {doc_name}{page_str}\n{text}")
    
    # Add image information
    if image_hits:
        context_parts.append("\n=== IMAGES ===")
        context_parts.append("The following images are provided. Examine them carefully.")
        
        # Collect image paths
        for hit in image_hits[:max_images]:
            if hit.get('image_path'):
                image_paths.append(hit['image_path'])
    
    # Add question
    context_parts.append(f"\n=== QUESTION ===\n{query}")
    context_parts.append("\n=== ANSWER ===")
    
    return "\n\n".join(context_parts), image_paths


def build_structured_prompt(query: str, structured_data: list) -> str:
    """
    Build a prompt for structured data results
    
    Args:
        query: User question
        structured_data: List of structured data records
    
    Returns:
        str: Formatted prompt
    """
    context_parts = []
    
    context_parts.append("=== STRUCTURED DATA RESULTS ===")
    
    if structured_data:
        for i, record in enumerate(structured_data, 1):
            context_parts.append(f"\nRecord {i}:")
            for key, value in record.items():
                if value is not None and str(value).strip():
                    context_parts.append(f"  • {key}: {value}")
    else:
        context_parts.append("No structured data found.")
    
    context_parts.append(f"\n=== QUESTION ===\n{query}")
    context_parts.append("\n=== ANSWER ===")
    
    return "\n".join(context_parts)


def build_hybrid_prompt(query: str, text_hits: list, structured_data: list = None, image_hits: list = None) -> str:
    """
    Build a prompt combining all available sources
    
    Args:
        query: User question
        text_hits: Retrieved text chunks
        structured_data: Structured data results
        image_hits: Retrieved image metadata
    
    Returns:
        str: Combined prompt
    """
    context_parts = []
    
    # Add structured data first (priority for numerical queries)
    if structured_data:
        context_parts.append("=== STRUCTURED DATA ===")
        for i, record in enumerate(structured_data[:5], 1):
            context_parts.append(f"\nRecord {i}:")
            for key, value in record.items():
                if value is not None and str(value).strip():
                    context_parts.append(f"  • {key}: {value}")
    
    # Add text context
    if text_hits:
        context_parts.append("\n=== DOCUMENT PASSAGES ===")
        for i, hit in enumerate(text_hits[:3], 1):
            doc_name = hit.get('doc_name', 'Unknown').replace('_', ' ').title()
            text = hit.get('text', '')[:600]
            page = hit.get('page_id', '')
            page_str = f" [Page {page}]" if page else ""
            context_parts.append(f"\n[{i}] {doc_name}{page_str}:\n{text}")
    
    # Add image descriptions
    if image_hits:
        context_parts.append("\n=== IMAGES ===")
        for i, hit in enumerate(image_hits[:2], 1):
            doc_name = hit.get('doc_name', 'Unknown').replace('_', ' ').title()
            desc = hit.get('text', '')[:300]
            context_parts.append(f"\n[Image {i}] {doc_name}: {desc}")
    
    # Add question
    context_parts.append(f"\n=== QUESTION ===\n{query}")
    context_parts.append("\n=== ANSWER ===")
    
    return "\n\n".join(context_parts)