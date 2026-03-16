import torch
import logging
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# Assuming config handles device selection
import config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ValidationModel:
    def __init__(self, model_path: str):
        self.device = config.DEVICE if hasattr(config, 'DEVICE') else ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading Validation Model onto {self.device}...")
        try:
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
            
            # CRITICAL: Move model to GPU and set to evaluation mode
            self.model = DistilBertForSequenceClassification.from_pretrained(model_path).to(self.device)
            self.model.eval() 
            
            logger.info("Validation Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load the validation model from {model_path}: {e}")
            raise

    def score(self, query: str, context: str) -> float:
        """
        Scores the relevance of a given context to the user's query.
        Returns a probability float between 0.0 and 1.0.
        """
        # Validate inputs to prevent tokenizer crashes
        if not query or not context:
            logger.warning("Empty query or context provided. Returning relevance score of 0.0.")
            return 0.0

        try:
            # CRITICAL: Explicitly set max_length to avoid out-of-bounds memory errors
            inputs = self.tokenizer(
                query,
                context,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device) # Move inputs to GPU

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

            # Extract the probability for class 1 (relevant)
            probs = torch.softmax(logits, dim=1)
            relevance_score = probs[0][1].item()
            
            return relevance_score

        except Exception as e:
            logger.error(f"Error during validation scoring for query '{query[:20]}...': {e}")
            # Fail safe: return 0.0 so bad context is rejected rather than crashing the app
            return 0.0