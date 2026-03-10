
import torch, transformers, sentence_transformers
print("torch:", getattr(torch, '__version__', None))
print("torch.cuda available:", torch.cuda.is_available())
print("transformers:", transformers.__version__)
print("sentence-transformers:", sentence_transformers.__version__)
# from transformers import CLIPProcessor, CLIPModel

# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# print("CLIP loaded successfully")