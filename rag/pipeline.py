import logging
from typing import List, Dict, Any

from rag.embedding_models import EmbeddingModels
from rag.retriever import Retriever
from rag.context_decision_engine import ContextDecisionEngine
from rag.validation_model import ValidationModel
from rag.reranker import Reranker
from rag.fusion import fuse_results
from rag.generator import Generator

# Set up logging to track the life of a query
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s')
logger = logging.getLogger(__name__)

class MultimodalRAG:
    def __init__(self):
        logger.info("Initializing the Multimodal RAG Pipeline. This may take a moment...")
        try:
            self.embedder = EmbeddingModels()
            self.retriever = Retriever()
            self.cde = ContextDecisionEngine("models/cde")
            self.validator = ValidationModel("models/validation")
            self.reranker = Reranker()
            self.generator = Generator()
            logger.info("All pipeline components loaded successfully and are ready for queries.")
        except Exception as e:
            logger.error(f"Critical failure during pipeline initialization: {e}")
            raise

    def run(self, query: str) -> str:
        """
        Executes the end-to-end multimodal RAG pipeline for a given query.
        """
        logger.info(f"--- Processing New Query: '{query}' ---")
        try:
            # 1. Routing: Decide what modalities we actually need
            modality = self.cde.predict(query)
            logger.info(f"Context Decision Engine routed query as: {modality.upper()}")

            # 2. Embedding: Convert query to vector
            q_vec = self.embedder.embed_text(query)

            # 3. Conditional Retrieval: Only fetch what the CDE recommended
            text_res, image_res = [], []
            
            if modality in ["text", "hybrid"]:
                logger.info("Retrieving text documents...")
                text_res = self.retriever.retrieve_text(q_vec)
                
            if modality in ["image", "hybrid"]:
                logger.info("Retrieving image documents...")
                image_res = self.retriever.retrieve_image(q_vec)

            # 4. Fusion: Combine and rank using RRF
            fused = fuse_results(text_res, image_res)
            
            if not fused:
                logger.warning("Retrieval returned no results. Bypassing validation and reranking.")
                return self.generator.generate(query, [])

            # 5. Validation: Filter out irrelevant context
            validated = []
            for item in fused:
                # Safely extract text for the validator. If it's an image, validate its description.
                payload = item.get("payload", {})
                context_text = payload.get("text", "") or payload.get("description", "")
                
                v_score = self.validator.score(query, context_text)
                
                if v_score > 0.5:
                    validated.append(item)

            if not validated:
                logger.warning("All retrieved contexts failed validation. Falling back to top 3 fused results.")
                validated = fused[:3] # Fallback so the LLM has *something* to work with

            # 6. Reranking: Cross-Encoder semantic sorting
            logger.info(f"Reranking {len(validated)} validated contexts...")
            ranked_contexts = self.reranker.rerank(query, validated)

            # 7. Generation: Build prompt and call LLM
            logger.info("Passing top contexts to the Generator...")
            answer = self.generator.generate(query, ranked_contexts)

            logger.info("Pipeline execution complete.")
            return answer

        except Exception as e:
            logger.error(f"Pipeline execution failed during query processing: {e}")
            return "I apologize, but an internal system error occurred while processing your admissions query."