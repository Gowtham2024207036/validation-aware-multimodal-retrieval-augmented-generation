import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from qdrant_utils import QdrantManager
import config

def main():

    qdrant = QdrantManager()

    # Create collections
    qdrant.create_collections()

    print("\nGenerating dummy vectors...")

    vectors = []
    payloads = []

    for i in range(10):

        vec = np.random.rand(config.TEXT_VECTOR_SIZE).tolist()

        vectors.append(vec)

        payloads.append({
            "chunk_id": i,
            "type": "text"
        })

    qdrant.insert_vectors(
        config.TEXT_COLLECTION,
        vectors,
        payloads
    )

    print("\nTesting retrieval...")

    query = np.random.rand(config.TEXT_VECTOR_SIZE).tolist()

    results = qdrant.search(
        config.TEXT_COLLECTION,
        query,
        config.TOP_K_TEXT
    )

    for r in results:
         print(r.payload, "score:", r.score)


if __name__ == "__main__":

    main()