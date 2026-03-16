"""
lmstudio_client.py
------------------
Client for LM Studio's local OpenAI-compatible API.
Supports:
  - Text-only chat
  - Multimodal chat (text + images) using Qwen2.5-VL vision capabilities
  - Streaming and non-streaming responses
  - Connection health check

LM Studio endpoints used:
  GET  /api/v1/models          → verify model is loaded
  POST /api/v1/chat/completions → generate response
"""

import os
import sys
import base64
import logging
import requests
from pathlib import Path

logger = logging.getLogger(__name__)

# LM Studio default settings — change if you've configured a different port
LMSTUDIO_BASE_URL = "http://localhost:1234"
MODEL_ID          = "qwen2.5-vl-7b-instruct"
DEFAULT_TIMEOUT   = 120   # seconds — VL models can be slow on first call

# Paths — adjust relative to your project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMAGES_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "raw"
)


def health_check() -> bool:
    """Check LM Studio is running and the model is loaded."""
    try:
        resp = requests.get(f"{LMSTUDIO_BASE_URL}/api/v1/models", timeout=5)
        if resp.status_code != 200:
            return False
        models = resp.json().get("data", [])
        loaded = [m.get("id", "") for m in models]
        if not any(MODEL_ID.lower() in m.lower() for m in loaded):
            logger.warning(
                "LM Studio running but model '%s' not loaded. "
                "Loaded models: %s", MODEL_ID, loaded
            )
            return False
        return True
    except requests.exceptions.ConnectionError:
        return False


def image_path_to_base64(image_path: str) -> str | None:
    """
    Convert a relative image path (from Qdrant payload) to base64.
    e.g. "images/COSTCO_2021_10K_image17.jpg" → base64 string
    """
    full_path = os.path.join(IMAGES_ROOT, image_path)
    if not os.path.exists(full_path):
        logger.warning("Image not found: %s", full_path)
        return None
    try:
        with open(full_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        logger.warning("Could not read image %s: %s", full_path, e)
        return None


def get_image_mime(image_path: str) -> str:
    """Infer MIME type from file extension."""
    ext = Path(image_path).suffix.lower()
    return {"jpg": "image/jpeg", ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg", ".png": "image/png",
            ".gif": "image/gif", ".webp": "image/webp"}.get(ext, "image/jpeg")


def build_multimodal_message(
    text_prompt: str,
    image_paths: list[str],
    max_images: int = 3,
) -> dict:
    """
    Build an OpenAI-format user message with text + images.
    Qwen2.5-VL accepts: {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
    """
    content = []

    # Add images first (VL models process images before the text question)
    for img_path in image_paths[:max_images]:
        b64 = image_path_to_base64(img_path)
        if b64 is None:
            continue
        mime = get_image_mime(img_path)
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime};base64,{b64}"
            }
        })

    # Add text prompt after images
    content.append({"type": "text", "text": text_prompt})

    return {"role": "user", "content": content}


def chat(
    messages: list[dict],
    temperature: float = 0.1,
    max_tokens: int = 512,
    stream: bool = False,
) -> str:
    """
    Send a chat request to LM Studio.
    Returns the assistant's response text.
    Low temperature (0.1) for factual financial QA — reduces hallucination.
    """
    payload = {
        "model":       MODEL_ID,
        "messages":    messages,
        "temperature": temperature,
        "max_tokens":  max_tokens,
        "stream":      stream,
    }

    try:
        resp = requests.post(
            f"{LMSTUDIO_BASE_URL}/api/v1/chat/completions",
            json=payload,
            timeout=DEFAULT_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    except requests.exceptions.ConnectionError:
        return "ERROR: Cannot connect to LM Studio. Is it running on localhost:1234?"
    except requests.exceptions.Timeout:
        return "ERROR: LM Studio timed out. The model may be slow — try reducing max_tokens."
    except requests.exceptions.HTTPError as e:
        return f"ERROR: LM Studio HTTP error: {e}"
    except (KeyError, IndexError) as e:
        return f"ERROR: Unexpected response format from LM Studio: {e}"
