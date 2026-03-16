from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from qdrant_client.http.exceptions import UnexpectedResponse
import logging
import config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        logger.info("Initializing Qdrant client...")
        try:
            self.client = QdrantClient(
                host=config.QDRANT_HOST,
                port=config.QDRANT_PORT,
                timeout=10.0  # Fails fast if the database is down
            )
            logger.info("Successfully connected to Qdrant.")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant at {config.QDRANT_HOST}:{config.QDRANT_PORT}. Error: {e}")
            raise

    def setup_collections(self):
        """
        Safely creates collections only if they do not already exist.
        """
        self._create_if_not_exists(config.TEXT_COLLECTION, config.TEXT_VECTOR_SIZE)
        self._create_if_not_exists(config.IMAGE_COLLECTION, config.IMAGE_VECTOR_SIZE)

    def _create_if_not_exists(self, collection_name: str, dimension: int):
        try:
            if not self.client.collection_exists(collection_name):
                logger.info(f"Creating missing collection: '{collection_name}'...")
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=dimension,
                        distance=Distance.COSINE
                    ),
                )
            else:
                logger.info(f"Collection '{collection_name}' already exists. Skipping creation.")
        except Exception as e:
            logger.error(f"Error checking or creating collection '{collection_name}': {e}")
            raise

    def upsert(self, collection: str, points: list, batch_size: int = 100):
        """
        Upserts points in batches to prevent timeouts and memory crashes.
        """
        if not points:
            logger.warning("No points provided for upsert.")
            return

        try:
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=collection,
                    points=batch
                )
                logger.debug(f"Upserted batch of {len(batch)} points to '{collection}'.")
            
            logger.info(f"Successfully upserted a total of {len(points)} points to '{collection}'.")
        except Exception as e:
            logger.error(f"Failed during upsert to '{collection}': {e}")
            raise

    def search(self, collection: str, vector: list, top_k: int = 5):
        """
        Queries the database for the nearest vectors.
        """
        try:
            results = self.client.search(
                collection_name=collection,
                query_vector=vector,
                limit=top_k
            )
            return results
        except Exception as e:
            logger.error(f"Search failed on collection '{collection}': {e}")
            raise