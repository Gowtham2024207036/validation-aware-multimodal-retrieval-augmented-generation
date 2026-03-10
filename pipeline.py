import fitz
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

import config


class MMDocRAG:

    def __init__(self):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print("Device:", self.device)

        # Text encoder
        self.text_encoder = SentenceTransformer(
            config.TEXT_MODEL,
            device=self.device
        )

        # Image encoder
        self.clip_model = CLIPModel.from_pretrained(
            config.IMAGE_MODEL
        ).to(self.device)

        self.clip_processor = CLIPProcessor.from_pretrained(
            config.IMAGE_MODEL
        )

        # Qdrant
        self.client = QdrantClient(
            host=config.QDRANT_HOST,
            port=config.QDRANT_PORT
        )

        self.initialize_collections()


    def initialize_collections(self):

        self.client.recreate_collection(
            collection_name=config.TEXT_COLLECTION,
            vectors_config=VectorParams(
                size=config.TEXT_VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )

        self.client.recreate_collection(
            collection_name=config.IMAGE_COLLECTION,
            vectors_config=VectorParams(
                size=config.IMAGE_VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )

        self.client.recreate_collection(
            collection_name=config.TABLE_COLLECTION,
            vectors_config=VectorParams(
                size=config.TEXT_VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )


    def parse_document(self, pdf_path):

        doc = fitz.open(pdf_path)

        texts = []
        images = []

        for page_num in range(len(doc)):

            page = doc.load_page(page_num)

            text = page.get_text()

            if text.strip():
                texts.append((page_num, text))

            image_list = page.get_images(full=True)

            for img_index, img in enumerate(image_list):

                xref = img[0]

                base_image = doc.extract_image(xref)

                image_bytes = base_image["image"]

                image = Image.open(
                    np.frombuffer(image_bytes, dtype=np.uint8)
                )

                images.append((page_num, image))

        return texts, images


    def embed_text(self, text):

        vec = self.text_encoder.encode(
            text,
            normalize_embeddings=True
        )

        return vec.tolist()


    def embed_image(self, image):

        inputs = self.clip_processor(
            images=image,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():

            emb = self.clip_model.get_image_features(**inputs)

        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)

        return emb.cpu().numpy()[0].tolist()


    def index_document(self, pdf_path):

        texts, images = self.parse_document(pdf_path)

        text_points = []
        image_points = []

        idx = 0

        for page, text in tqdm(texts):

            vec = self.embed_text(text)

            text_points.append(
                PointStruct(
                    id=idx,
                    vector=vec,
                    payload={
                        "modality": "text",
                        "page": page
                    }
                )
            )

            idx += 1

        for page, img in tqdm(images):

            vec = self.embed_image(img)

            image_points.append(
                PointStruct(
                    id=idx,
                    vector=vec,
                    payload={
                        "modality": "image",
                        "page": page
                    }
                )
            )

            idx += 1

        self.client.upsert(
            collection_name=config.TEXT_COLLECTION,
            points=text_points
        )

        self.client.upsert(
            collection_name=config.IMAGE_COLLECTION,
            points=image_points
        )


    def context_decision_engine(self, question):

        question = question.lower()

        if "table" in question or "value" in question or "how many" in question:

            return {
                "text": 0.3,
                "image": 0.2,
                "table": 0.5
            }

        if "figure" in question or "graph" in question:

            return {
                "text": 0.2,
                "image": 0.7,
                "table": 0.1
            }

        return {
            "text": 0.7,
            "image": 0.2,
            "table": 0.1
        }


    def retrieve(self, question):

        query_vec = self.embed_text(question)

        text_hits = self.client.search(
            collection_name=config.TEXT_COLLECTION,
            query_vector=query_vec,
            limit=config.TOP_K_TEXT
        )

        image_hits = self.client.search(
            collection_name=config.IMAGE_COLLECTION,
            query_vector=query_vec,
            limit=config.TOP_K_IMAGE
        )

        return text_hits, image_hits