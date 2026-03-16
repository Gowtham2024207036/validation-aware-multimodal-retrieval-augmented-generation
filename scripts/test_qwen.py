import base64
import requests
import json
import os

# LM Studio local server URL (Default is usually port 1234)
API_URL = "http://localhost:1234/v1/chat/completions"
MODEL_ID = "qwen2.5-vl-7b-instruct"

def encode_image(image_path):
    """Encodes an image to Base64 to send over the API."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def test_qwen_generation():
    print(f"🚀 Connecting to LM Studio ({MODEL_ID})...")
    
    # The actual Costco image from your dataset
    # We are manually feeding it the right image just to test its math skills
    image_path = os.path.join("data", "raw", "images", "COSTCO_2021_10K_image5.jpg")
    
    if not os.path.exists(image_path):
        print(f"❌ Error: Cannot find {image_path}. Please check the filename.")
        return

    print("🖼️  Encoding Costco Balance Sheet image...")
    base64_image = encode_image(image_path)
    
    test_query = "What is the Long-term Debt to Total Liabilities for COSTCO in FY2021? Round your answer to two decimal places. Show your math."
    print(f"\n🗣️  USER QUERY: '{test_query}'")
    
    # OpenAI-compatible payload structure for Vision Models
    payload = {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": test_query
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "temperature": 0.1, # Keep it low for factual math
        "max_tokens": 500
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    print("\n🧠 Sending image and question to Qwen2.5-VL... (Waiting for inference)")
    
    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        
        result = response.json()
        answer = result['choices'][0]['message']['content']
        
        print("\n==================================================")
        print("🤖 QWEN2.5-VL RESPONSE:")
        print("==================================================")
        print(answer)
        print("==================================================")
        
    except requests.exceptions.RequestException as e:
        print(f"\n❌ API Connection Error: {e}")
        print("Make sure LM Studio 'Local Server' is started and running on port 1234.")

if __name__ == "__main__":
    test_qwen_generation()