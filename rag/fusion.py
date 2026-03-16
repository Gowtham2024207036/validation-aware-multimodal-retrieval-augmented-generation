import logging
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fuse_results(text_results: List[Dict[str, Any]], image_results: List[Dict[str, Any]], top_k: int = 5, rrf_k: int = 60) -> List[Dict[str, Any]]:
    """
    Combines text and image retrieval results using Reciprocal Rank Fusion (RRF).
    This prevents score imbalance between different embedding models.
    """
    # Fail-safes for empty lists to prevent iteration crashes
    text_results = text_results or []
    image_results = image_results or []
    
    if not text_results and not image_results:
        logger.warning("Both text and image result lists are empty. Returning empty fusion list.")
        return []

    # RRF Dictionary to accumulate scores for each unique item
    # Using the payload's quote_id or the Qdrant ID to track uniqueness
    fused_scores = {}
    
    # Helper function to process and rank lists
    def apply_rrf(results: List[Dict[str, Any]], modality: str):
        for rank, item in enumerate(results):
            # Extract safe dictionary keys
            item_id = item.get("id", f"unknown_{modality}_{rank}")
            
            if item_id not in fused_scores:
                # Store the full original item so we don't lose the img_path or page_id
                fused_scores[item_id] = {
                    "item": item,
                    "rrf_score": 0.0
                }
            
            # RRF Formula: 1 / (k + rank)
            # We add 1 to rank because enumerate starts at 0
            fused_scores[item_id]["rrf_score"] += 1.0 / (rrf_k + rank + 1)

    # Apply RRF to both result sets
    apply_rrf(text_results, "text")
    apply_rrf(image_results, "image")

    # Sort the combined items by their new RRF score in descending order
    sorted_fused = sorted(
        fused_scores.values(),
        key=lambda x: x["rrf_score"],
        reverse=True
    )

    # Extract just the original dictionary items (with the new score attached if desired)
    final_combined = []
    for rank_data in sorted_fused[:top_k]:
        final_item = rank_data["item"]
        final_item["fusion_score"] = rank_data["rrf_score"]  # Append for debugging
        final_combined.append(final_item)

    logger.info(f"Successfully fused {len(text_results)} text and {len(image_results)} image results into {len(final_combined)} top items.")
    return final_combined