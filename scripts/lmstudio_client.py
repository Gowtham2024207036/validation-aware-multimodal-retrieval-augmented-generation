"""
lmstudio_client.py
------------------
Client for LM Studio's local OpenAI-compatible API.
Endpoint confirmed working: POST /v1/chat/completions

Public API:
    check_connection()                              -> bool
    generate(messages)                              -> str
    generate_text_only(system, user_text)           -> str
    generate_with_images(system, user_text,
                         image_paths, images_root)  -> str
"""

import os
import sys
import io
import base64
import logging
import requests
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234")
LM_STUDIO_MODEL    = os.getenv("LM_STUDIO_MODEL",    "qwen2.5-vl-7b-instruct")
LM_STUDIO_TIMEOUT  = int(os.getenv("LM_STUDIO_TIMEOUT", "120"))

# Confirmed working endpoint
CHAT_URL   = f"{LM_STUDIO_BASE_URL}/v1/chat/completions"
MODELS_URL = f"{LM_STUDIO_BASE_URL}/api/v1/models"


# ------------------------------------------------------------------ #
# Connection check
# ------------------------------------------------------------------ #

def check_connection() -> bool:
    """Return True if LM Studio is reachable."""
    try:
        r = requests.get(MODELS_URL, timeout=5)
        return r.status_code == 200
    except Exception:
        return False

# Alias
health_check = check_connection


# ------------------------------------------------------------------ #
# Image helper
# ------------------------------------------------------------------ #

def image_to_base64(image_path: str, images_root: str = "",
                    max_size: int = 512) -> str | None:
    """
    Load image, resize to max_size on longest edge, return base64 JPEG data URI.
    Returns None if file not found or cannot be read.
    max_size=512 keeps payload small — Qwen2.5-VL-7B has limited context.
    """
    full = os.path.join(images_root, image_path) if images_root else image_path
    if not os.path.exists(full):
        logger.warning("Image not found: %s", full)
        return None
    try:
        from PIL import Image as PILImage
        img = PILImage.open(full).convert("RGB")
        w, h = img.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            img   = img.resize((int(w * scale), int(h * scale)), PILImage.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"
    except Exception as e:
        logger.warning("Cannot encode image %s: %s", full, e)
        return None

# Alias
image_path_to_base64 = image_to_base64


# ------------------------------------------------------------------ #
# Core generate — text only messages
# ------------------------------------------------------------------ #

def generate(
    messages:    list[dict],
    max_tokens:  int   = 1024,
    temperature: float = 0.1,
) -> str:
    """
    Send messages to LM Studio /v1/chat/completions.
    All message content must be strings (not arrays) for text-only calls.
    Returns the assistant reply string. Never returns None.
    """
    payload = {
        "model":       LM_STUDIO_MODEL,
        "messages":    messages,
        "max_tokens":  max_tokens,
        "temperature": temperature,
        "stream":      False,
    }
    try:
        r = requests.post(
            CHAT_URL, json=payload,
            headers={"Content-Type": "application/json"},
            timeout=LM_STUDIO_TIMEOUT,
        )
        r.raise_for_status()
        reply = r.json()["choices"][0]["message"]["content"]
        return reply.strip() if reply else "No response from model."
    except requests.exceptions.ConnectionError:
        return "ERROR: Cannot connect to LM Studio. Is it running on localhost:1234?"
    except requests.exceptions.Timeout:
        return f"ERROR: LM Studio timed out after {LM_STUDIO_TIMEOUT}s."
    except requests.exceptions.HTTPError as e:
        # Log response body for debugging
        try:
            body = r.json()
            logger.error("LM Studio 400 body: %s", body)
        except Exception:
            pass
        return f"ERROR: HTTP {e}"
    except (KeyError, IndexError, TypeError) as e:
        return f"ERROR: Unexpected response format: {e}"
    except Exception as e:
        return f"ERROR: {e}"


# Alias for older code
def chat(messages, temperature=0.1, max_tokens=512) -> str:
    return generate(messages, max_tokens=max_tokens, temperature=temperature)


# ------------------------------------------------------------------ #
# Text-only generation
# ------------------------------------------------------------------ #

def generate_text_only(
    system_prompt: str,
    user_text:     str,
    max_tokens:    int   = 1024,
    temperature:   float = 0.1,
) -> str:
    """Pure text generation — no images. Fast and reliable."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_text},
    ]
    return generate(messages, max_tokens, temperature)


# ------------------------------------------------------------------ #
# Multimodal generation
# ------------------------------------------------------------------ #

def generate_with_images(
    system_prompt: str,
    user_text:     str,
    image_paths:   list[str],
    images_root:   str   = "",
    max_tokens:    int   = 1024,
    temperature:   float = 0.1,
    max_images:    int   = 1,
) -> str:
    """
    Multimodal generation with Qwen2.5-VL via LM Studio.

    Key facts about LM Studio + Qwen2.5-VL:
    - /v1/chat/completions is the correct endpoint
    - Content must be an array when images are included
    - System message must also use array format (not string) when user uses array
    - Limit to 1 image per request to avoid context overflow on 7B model
    - System prompt merged into user message to avoid mixed-format rejection
    """
    # Load images — only first max_images to avoid token overflow
    image_uris = []
    for p in (image_paths or [])[:max_images]:
        uri = image_to_base64(p, images_root)
        if uri:
            image_uris.append(uri)
            logger.debug("Loaded image: %s", p)

    # No images available — fall back to text only silently
    if not image_uris:
        logger.info("No images loaded — using text-only generation.")
        return generate_text_only(system_prompt, user_text, max_tokens, temperature)

    # Build multimodal content array
    # Merge system prompt into user text to avoid mixed string/array content types
    full_text = f"{system_prompt}\n\n{user_text}"
    content   = []
    for uri in image_uris:
        content.append({
            "type":      "image_url",
            "image_url": {"url": uri},
        })
    content.append({"type": "text", "text": full_text})

    messages = [
        {"role": "user", "content": content}
    ]

    result = generate(messages, max_tokens, temperature)

    # If multimodal failed, retry as text only
    if result.startswith("ERROR:"):
        logger.warning("Multimodal failed (%s), retrying as text-only.", result)
        return generate_text_only(system_prompt, user_text, max_tokens, temperature)

    return result