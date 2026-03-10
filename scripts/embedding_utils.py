# scripts/embedding_utils.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
from typing import List, Tuple
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

import config

# SentenceTransformer for text
from sentence_transformers import SentenceTransformer

# CLIP from transformers
from transformers import CLIPProcessor, CLIPModel


# ---------- Model loader (singleton-like) ----------
class _Models:
    text_model = None
    clip_model = None
    clip_processor = None
    device = config.DEVICE

def load_text_model():
    if _Models.text_model is None:
        print(f"Loading text model `{config.TEXT_MODEL}` on {_Models.device} ...")
        _Models.text_model = SentenceTransformer(config.TEXT_MODEL, device=_Models.device)
        print("Text model loaded.")
    return _Models.text_model

def load_clip_model():
    if _Models.clip_model is None or _Models.clip_processor is None:
        print(f"Loading CLIP model `{config.IMAGE_MODEL}` on {_Models.device} ...")
        _Models.clip_model = CLIPModel.from_pretrained(config.IMAGE_MODEL).to(_Models.device)
        _Models.clip_processor = CLIPProcessor.from_pretrained(config.IMAGE_MODEL)
        print("CLIP loaded.")
    return _Models.clip_model, _Models.clip_processor

# ---------- Utilities ----------
def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, ord=2, axis=-1, keepdims=True)
    norm[norm == 0] = 1.0
    return x / norm

# ---------- Embedding functions ----------
def embed_texts(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """
    Embed a list of texts and return list of vector lists (float).
    """
    model = load_text_model()
    vectors = []
    # sentence-transformers model.encode supports batching; we call it directly for efficiency.
    # normalize_embeddings=True gives unit vectors
    print(f"Embedding {len(texts)} texts (batch_size={batch_size}) ...")
    embeddings = model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)
    # embeddings is numpy array
    if isinstance(embeddings, np.ndarray):
        vectors = embeddings.tolist()
    else:
        vectors = [e.tolist() for e in embeddings]
    print("Text embedding complete.")
    return vectors


def embed_images(images: List[Image.Image], batch_size: int = 8) -> List[List[float]]:
    """
    Embed a list of PIL.Image objects and return list of vector lists (float).
    Uses CLIPModel.get_image_features and L2-normalizes results.
    """
    clip_model, clip_processor = load_clip_model()
    device = _Models.device
    vectors = []

    total = len(images)
    print(f"Embedding {total} images (batch_size={batch_size}) ...")

    # Process in batches
    for i in range(0, total, batch_size):
        batch_imgs = images[i : i + batch_size]
        # Prepare inputs
        inputs = clip_processor(images=batch_imgs, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            # Autocast to fp16 if cuda for speed/memory
            if device.startswith("cuda"):
                with torch.cuda.amp.autocast():
                    feats = clip_model.get_image_features(**inputs)
            else:
                feats = clip_model.get_image_features(**inputs)
            # move to cpu numpy
            feats = feats.cpu().numpy()
            # normalize
            feats = _l2_normalize(feats)
            for f in feats:
                vectors.append(f.tolist())

    print("Image embedding complete.")
    return vectors


# ---------- Small helpers for quick use ----------
def embed_text_single(text: str) -> List[float]:
    return embed_texts([text], batch_size=1)[0]

def embed_image_single(img: Image.Image) -> List[float]:
    return embed_images([img], batch_size=1)[0]