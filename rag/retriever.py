import logging
from typing import List, Dict, Any
from rag.vector_store import VectorStore
import config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Retriever:
    def __init__(self):
        try:
            self.db = VectorStore()
            logger.info("Retriever initialized and connected to VectorStore.")
        except Exception as e:
            logger.error(f"Failed to initialize VectorStore in Retriever: {e}")
            raise

    def _format_and_filter_results(self, raw_results: list, score_threshold: float) -> List[Dict[str, Any]]:
        """
        Parses Qdrant ScoredPoint objects into clean dictionaries and filters out bad matches.
        """
        formatted_results = []
        for point in raw_results:
            if point.score >= score_threshold:
                formatted_results.append({
                    "id": point.id,
                    "score": point.score,
                    "payload": point.payload  # This contains your text, page_id, or img_path
                })
            else:
                # Useful for debugging if your threshold is too strict
                logger.debug(f"Filtered out result. Score {point.score} is below threshold {score_threshold}")
                
        return formatted_results

    def retrieve_text(self, vector: list, top_k: int = None, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Retrieves text documents based on vector similarity, returning clean payloads.
        """
        limit = top_k or config.TOP_K
        try:
            raw_results = self.db.search(
                collection=config.TEXT_COLLECTION,
                vector=vector,
                top_k=limit
            )
            return self._format_and_filter_results(raw_results, threshold)
        except Exception as e:
            logger.error(f"Text retrieval failed: {e}")
            return []  # Return empty list to prevent application crash

    def retrieve_image(self, vector: list, top_k: int = None, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Retrieves image metadata based on vector similarity, returning clean payloads.
        """
        limit = top_k or config.TOP_K
        try:
            raw_results = self.db.search(
                collection=config.IMAGE_COLLECTION,
                vector=vector,
                top_k=limit
            )
            return self._format_and_filter_results(raw_results, threshold)
        except Exception as e:
            logger.error(f"Image retrieval failed: {e}")
            return []