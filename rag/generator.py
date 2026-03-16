import logging
import base64
import requests
from typing import List, Dict, Any
import config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Generator:
    def __init__(self, api_url: str = "http://localhost:1234/v1/chat/completions"):
        # Default LM Studio local server endpoint
        self.api_url = api_url
        logger.info(f"Initializing Generator, pointing to local LM Studio at {self.api_url}")

    def _encode_image(self, image_path: str) -> str:
        """Converts an image file to a Base64 string for the API payload."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to encode image at {image_path}: {e}")
            return ""

    def generate(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        """
        Builds a multimodal payload and sends it to the LM Studio server.
        """
        text_blocks = []
        base64_images = []

        # 1. Extract context text and encode images
        if not contexts:
            logger.warning("No context provided. The VLM will rely entirely on its internal knowledge.")
        else:
            for i, ctx in enumerate(contexts):
                if "text" in ctx:
                    text_blocks.append(f"[Document {i+1}]: {ctx['text']}")
                
                if "description" in ctx:
                    text_blocks.append(f"[Image {i+1} Description]: {ctx['description']}")
                    
                if "img_path" in ctx:
                    full_path = f"{config.DATA_DIR}/{ctx['img_path']}"
                    encoded_img = self._encode_image(full_path)
                    if encoded_img:
                        base64_images.append(encoded_img)
                        text_blocks.append(f"[Refer to the provided Image {i+1} for visual context]")

        context_string = "\n".join(text_blocks)
        
        # 2. Build the System Prompt
        # 2. Build the System Prompt
        system_prompt = (
            "You are an expert university admissions assistant. Your task is to answer the user's question "
            "using ONLY the provided text documents and images.\n\n"
            "You MUST follow these three rules:\n"
            "1. EVIDENCE: Base your answer strictly on the provided context. If the answer is not in the context, say 'I do not have enough information.'\n"
            "2. REASONING: Explain your logic step-by-step before giving the final answer. Show how you arrived at your conclusion.\n"
            "3. CITATION: You must cite the source of your information. When using text, append the document number like [Document 1]. "
            "When referencing an image, use the markdown format ![description](Image 1)."
        )

        # 3. Build the User Content Array (Text + Images)
        user_content = [
            {
                "type": "text",
                "text": f"Question: {query}\n\nContext:\n{context_string}\n\nAnswer:"
            }
        ]

        # Append each retrieved image to the payload
        for b64_img in base64_images:
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64_img}"
                }
            })

        # 4. Construct the final API Payload
        payload = {
            "model": "local-model", # LM Studio ignores this field, but requires it to be present
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.2, # Low temperature for factual RAG tasks
            "max_tokens": 1024
        }

        # 5. Send the Request to LM Studio
        logger.info(f"Sending query to LM Studio with {len(base64_images)} supporting images...")
        try:
            response = requests.post(self.api_url, json=payload, timeout=120)
            response.raise_for_status()
            
            response_data = response.json()
            final_answer = response_data["choices"][0]["message"]["content"]
            
            return final_answer

        except requests.exceptions.ConnectionError:
            logger.error("Could not connect to LM Studio. Did you click 'Start Server' on port 1234?")
            return "Error: Could not connect to the local VLM server. Please ensure LM Studio is running."
        except Exception as e:
            logger.error(f"Failed to generate response from LM Studio: {e}")
            return "I apologize, but an internal error occurred while generating the answer."