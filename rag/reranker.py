import logging
from typing import List, Dict, Any, Union
from sentence_transformers import CrossEncoder

# Assuming config handles device selection and model names
import config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Reranker:
    def __init__(self):
        # Fallback to the MS MARCO model if not defined in config
        model_name = getattr(config, 'RERANKER_MODEL', "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.device = getattr(config, 'DEVICE', 'cpu')
        
        logger.info(f"Loading Reranker model '{model_name}' onto {self.device}...")
        try:
            # CRITICAL: Pass the device to leverage GPU acceleration
            self.model = CrossEncoder(
                model_name,
                device=self.device
            )
            logger.info("Reranker loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Reranker model: {e}")
            raise

    def rerank(self, query: str, contexts: List[Union[str, Dict[str, Any]]], top_k: int = 3) -> List[Any]:
        """
        Reranks contexts based on their relevance to the query.
        Safely handles both raw strings and dictionary payloads from the Retriever.
        """
        if not query or not contexts:
            logger.warning("Empty query or contexts provided to Reranker. Returning original contexts.")
            return contexts

        try:
            # CRITICAL: Extract text from payloads if contexts are dictionaries
            text_contexts = []
            for c in contexts:
                if isinstance(c, dict) and "payload" in c and "text" in c["payload"]:
                    text_contexts.append(c["payload"]["text"])
                elif isinstance(c, str):
                    text_contexts.append(c)
                else:
                    logger.warning(f"Unrecognized context format in Reranker: {type(c)}. Skipping.")
                    text_contexts.append("") # Pad to maintain index alignment

            # Prepare pairs for the CrossEncoder
            pairs = [[query, text] for text in text_contexts]

            # Predict semantic similarity scores
            scores = self.model.predict(pairs)

            # Re-associate the scores with the ORIGINAL context objects (preserving metadata)
            ranked = sorted(
                zip(contexts, scores),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Return the original objects (now sorted), sliced to top_k
            return [item[0] for item in ranked[:top_k]]

        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            # Fail safe: return the original contexts un-reranked to keep the pipeline alive
            return contexts[:top_k]