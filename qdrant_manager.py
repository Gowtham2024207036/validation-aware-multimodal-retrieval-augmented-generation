from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import config


class QdrantManager:

    def __init__(self):

        print("Connecting to Qdrant...")

        self.client = QdrantClient(
            host=config.QDRANT_HOST,
            port=config.QDRANT_PORT
        )

        print("Connected to Qdrant")

    # ---------------------------------------
    # Create collections
    # ---------------------------------------

    def create_collections(self):

        print("Creating collections...")

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

        print("Collections created successfully")

    # ---------------------------------------
    # Insert vectors
    # ---------------------------------------

    def insert_vectors(self, collection_name, vectors, payloads):

        points = []

        for i, vec in enumerate(vectors):

            points.append(
                PointStruct(
                    id=i,
                    vector=vec,
                    payload=payloads[i]
                )
            )

        self.client.upsert(
            collection_name=collection_name,
            points=points
        )

        print(f"{len(points)} vectors inserted into {collection_name}")

    # ---------------------------------------
    # Search vectors
    # ---------------------------------------

    def search(self, collection_name, query_vector, top_k):

        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k
        )

        return results
