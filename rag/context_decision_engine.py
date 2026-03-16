import torch
import logging
from typing import Union, List, Dict, Any
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# Assuming config handles device selection (e.g., config.DEVICE = 'cuda')
import config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ContextDecisionEngine:
    def __init__(self, model_path: str):
        self.device = config.DEVICE if hasattr(config, 'DEVICE') else ("cuda" if torch.cuda.is_available() else "cpu")
        self.labels = ["text", "image", "table", "hybrid"]
        
        logger.info(f"Loading Context Decision Engine onto {self.device}...")
        try:
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
            
            # CRITICAL: Move model to GPU and set to evaluation mode
            self.model = DistilBertForSequenceClassification.from_pretrained(model_path).to(self.device)
            self.model.eval() 
            
            logger.info("Decision Engine loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load the model or tokenizer from {model_path}: {e}")
            raise

    def predict(self, query: Union[str, List[str]], threshold: float = 0.5) -> Union[str, List[Dict[str, Any]]]:
        """
        Predicts the context type required for the query, returning the label.
        Handles both single queries and batches.
        """
        if not query:
            logger.warning("Received empty query. Defaulting to 'text'.")
            return "text"

        is_single = isinstance(query, str)
        queries = [query] if is_single else query

        try:
            # CRITICAL: Add truncation and padding to prevent crashes on long inputs
            inputs = self.tokenizer(
                queries,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(self.device) # Move inputs to GPU

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
            # Calculate probabilities to gauge confidence
            probs = torch.softmax(logits, dim=-1)
            
            results = []
            for i in range(len(queries)):
                max_prob, pred_idx = torch.max(probs[i], dim=0)
                label = self.labels[pred_idx.item()]
                
                # Fallback logic if confidence is too low
                if max_prob.item() < threshold:
                    logger.debug(f"Low confidence ({max_prob.item():.2f}) for query. Defaulting to 'hybrid'.")
                    label = "hybrid"
                    
                results.append({
                    "query": queries[i],
                    "predicted_type": label,
                    "confidence": max_prob.item()
                })

            # Return just the string label for a single query to maintain your original API
            return results[0]["predicted_type"] if is_single else results

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            # Safe fallback for production to keep the pipeline moving
            return "text" if is_single else [{"query": q, "predicted_type": "text"} for q in queries]