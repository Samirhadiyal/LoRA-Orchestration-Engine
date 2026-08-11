import logging

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.core.config import settings

COLLECTION_NAME = "neuromesh_knowledge"
VECTOR_SIZE = 384

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def init_collection() -> None:
    client = QdrantClient(url=settings.QDRANT_URL, check_compatibility=False)
    collections = client.get_collections().collections

    if any(collection.name == COLLECTION_NAME for collection in collections):
        logger.info("Qdrant collection '%s' already exists.", COLLECTION_NAME)
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    logger.info(
        "Created Qdrant collection '%s' with vector size %s.",
        COLLECTION_NAME,
        VECTOR_SIZE,
    )


if __name__ == "__main__":
    init_collection()
