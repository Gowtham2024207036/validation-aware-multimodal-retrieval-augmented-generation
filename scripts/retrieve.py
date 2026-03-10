import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from embedding_utils import embed_text_single
from qdrant_utils import QdrantManager
from PIL import Image
from typing import List, Dict, Any

from transformers import CLIPProcessor, CLIPModel
import torch


class Retriever:
    """
    Multimodal retriever supporting:
    • text → text retrieval
    • text → image retrieval
    • image → image retrieval
    """

    def __init__(self):
        try:
            self.qdrant = QdrantManager()
            print("✓ Retriever initialized")

            # load CLIP once for image search
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.clip_model = CLIPModel.from_pretrained(config.IMAGE_MODEL).to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained(config.IMAGE_MODEL)

        except Exception as e:
            print(f"❌ Error initializing retriever: {e}")
            raise

    # --------------------------------------------------
    # TEXT → TEXT SEARCH
    # --------------------------------------------------
    def search_text(self, query: str, top_k: int = None) -> List[Any]:

        top_k = top_k or config.TOP_K_TEXT

        try:
            query_vector = embed_text_single(query)

            results = self.qdrant.search(
                collection_name=config.TEXT_COLLECTION,
                query_vector=query_vector,
                top_k=top_k
            )

            return results

        except Exception as e:
            print(f"❌ Error searching text: {e}")
            raise

    # --------------------------------------------------
    # TEXT → IMAGE SEARCH (CLIP text encoder)
    # --------------------------------------------------
    def search_images_by_text(self, query: str, top_k: int = None) -> List[Any]:

        top_k = top_k or config.TOP_K_IMAGE

        try:
            inputs = self.clip_processor(
                text=[query],
                return_tensors="pt",
                padding=True
            ).to(self.device)

            with torch.no_grad():
                text_features = self.clip_model.get_text_features(**inputs)

            query_vector = text_features[0].cpu().numpy().tolist()

            results = self.qdrant.search(
                collection_name=config.IMAGE_COLLECTION,
                query_vector=query_vector,
                top_k=top_k
            )

            return results

        except Exception as e:
            print(f"❌ Error searching images by text: {e}")
            raise

    # --------------------------------------------------
    # IMAGE → IMAGE SEARCH
    # --------------------------------------------------
    def search_images_by_image(self, image_path: str, top_k: int = None) -> List[Any]:

        top_k = top_k or config.TOP_K_IMAGE

        try:
            image = Image.open(image_path).convert("RGB")

            inputs = self.clip_processor(
                images=image,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)

            query_vector = image_features[0].cpu().numpy().tolist()

            results = self.qdrant.search(
                collection_name=config.IMAGE_COLLECTION,
                query_vector=query_vector,
                top_k=top_k
            )

            return results

        except Exception as e:
            print(f"❌ Error searching images by image: {e}")
            raise

    # --------------------------------------------------
    # MULTIMODAL SEARCH
    # --------------------------------------------------
    def search_all(self, query: str) -> Dict[str, List[Any]]:

        text_results = self.search_text(query)
        image_results = self.search_images_by_text(query)

        return {
            "text": text_results,
            "images": image_results
        }


if __name__ == "__main__":

    retriever = Retriever()

    query = "What does the architecture diagram show?"

    results = retriever.search_all(query)

    print("\nText results:", len(results["text"]))
    print("Image results:", len(results["images"]))