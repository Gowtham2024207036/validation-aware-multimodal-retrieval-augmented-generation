# scripts/test_embedding.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from embedding_utils import embed_texts, embed_images, embed_text_single, embed_image_single
import config
from PIL import Image
import glob
import json
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

print("\nDEBUG INFO")
print("Document folder path:", config.DOCUMENT_FOLDER)
print("Absolute path:", os.path.abspath(config.DOCUMENT_FOLDER))

if os.path.exists(config.DOCUMENT_FOLDER):
    print("Folder exists")
    print("Files inside folder:")
    all_files = os.listdir(config.DOCUMENT_FOLDER)
    print(all_files)
else:
    print("Folder NOT FOUND")
    print("Creating folder:", config.DOCUMENT_FOLDER)
    os.makedirs(config.DOCUMENT_FOLDER, exist_ok=True)


def test_texts():
    samples = [
        "What is the long-term debt to total liabilities for Costco in FY2021?",
        "Describe the trend shown in the revenue graph.",
        "How many satellites were launched in 2014?",
        "Summarize the key finding from the executive summary.",
        "What is the contribution profit from memberships in FY2015?"
    ]
    vecs = embed_texts(samples, batch_size=8)
    print("\n--- Text Embedding Results ---")
    for i, v in enumerate(vecs):
        v_np = np.array(v, dtype=float)
        print(f"Text #{i+1}: dim={v_np.shape[0]}, L2norm={np.linalg.norm(v_np):.6f}")


def test_images():
    # look for common image files under sample_docs
    image_folder = config.DOCUMENT_FOLDER
    imgs = []
    img_names = []
    
    print(f"\n--- Searching for images in: {image_folder} ---")
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(image_folder):
        print(f"Checking directory: {root}")
        print(f"Found {len(files)} files")
        
        for file in files:
            # Check for image extensions (case-insensitive)
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif")):
                img_path = os.path.join(root, file)
                print(f"Found image: {file}")
                try:
                    img = Image.open(img_path).convert("RGB")
                    imgs.append(img)
                    img_names.append(file)
                    print(f"  ✓ Successfully loaded: {file} (size: {img.size})")
                except Exception as e:
                    print(f"  ✗ Error loading {file}: {e}")
    
    # if no standalone images found
    if not imgs:
        print("\nNo standalone images found in sample_docs/.")
        print("If you have PDFs, use the document parser to extract images.")
        print("\nTip: Make sure image files have extensions: .png, .jpg, .jpeg, .bmp, .tiff")
    else:
        print(f"\n--- Found {len(imgs)} images, testing embedding ---")
        n = min(len(imgs), 3)
        imgs_to_test = imgs[:n]
        names_to_test = img_names[:n]
        
        vecs = embed_images(imgs_to_test, batch_size=2)
        print("\n--- Image Embedding Results ---")
        for i, v in enumerate(vecs):
            v_np = np.array(v, dtype=float)
            print(f"Image #{i+1} ({names_to_test[i]}): dim={v_np.shape[0]}, L2norm={np.linalg.norm(v_np):.6f}")


if __name__ == "__main__":
    print("Running embedding tests...")
    test_texts()
    test_images()
    print("\nEmbedding test complete.")