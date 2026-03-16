import torch
from PIL import Image, UnidentifiedImageError
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import logging
import os
from typing import List, Union

# Assuming config is a local module
import config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmbeddingModels:
    def __init__(self):
        try:
            logger.info("Loading text embedding model...")
            self.text_model = SentenceTransformer(
                config.TEXT_MODEL,
                device=config.DEVICE
            )

            logger.info("Loading CLIP model and processor...")
            self.clip_model = CLIPModel.from_pretrained(
                config.IMAGE_MODEL
            ).to(config.DEVICE)
            
            self.clip_processor = CLIPProcessor.from_pretrained(
                config.IMAGE_MODEL
            )
            logger.info("All embedding models loaded successfully.")
            
        except Exception as e:
            logger.error(f"Failed to load embedding models: {e}")
            raise

    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Embeds a single string or a list of strings into a normalized vector.
        """
        if not text:
            logger.warning("Received empty text for embedding.")
            return np.array([])

        try:
            # SentenceTransformer handles batching natively if a list is passed
            vecs = self.text_model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False 
            )
            return vecs
        except Exception as e:
            logger.error(f"Error embedding text: {e}")
            raise

    def embed_image(self, image_path: str) -> np.ndarray:
        """
        Embeds a single image from a filepath into a normalized vector.
        """
        if not os.path.exists(image_path):
            logger.error(f"Image not found at path: {image_path}")
            raise FileNotFoundError(f"Image not found at path: {image_path}")

        try:
            image = Image.open(image_path).convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            logger.error(f"Failed to open or identify image at {image_path}: {e}")
            raise

        try:
            inputs = self.clip_processor(
                images=image,
                return_tensors="pt"
            ).to(config.DEVICE)

            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)

            # Convert to numpy, extract the first element, and normalize
            vec = image_features.cpu().numpy()[0]
            
            # Avoid division by zero in the extremely rare case of a zero vector
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
                
            return vec
            
        except Exception as e:
            logger.error(f"Error during image feature extraction for {image_path}: {e}")
            raise
            
    # Optional but highly recommended: Add a batch image processing method
    def embed_images_batch(self, image_paths: List[str]) -> np.ndarray:
        """
        Embeds a batch of images to fully utilize GPU parallelization.
        """
        valid_images = []
        for path in image_paths:
            try:
                img = Image.open(path).convert("RGB")
                valid_images.append(img)
            except Exception as e:
                logger.warning(f"Skipping unreadable image {path}: {e}")
                
        if not valid_images:
            return np.array([])
            
        inputs = self.clip_processor(images=valid_images, return_tensors="pt").to(config.DEVICE)
        
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
            
        vecs = image_features.cpu().numpy()
        # Normalize along the feature axis
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = np.divide(vecs, norms, out=np.zeros_like(vecs), where=norms!=0)
        
        return vecs