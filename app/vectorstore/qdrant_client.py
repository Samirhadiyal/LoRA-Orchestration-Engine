# app/vectorstore/qdrant_client.py
from qdrant_client import AsyncQdrantClient

from app.core.config import settings

# Initialize the global async client
client = AsyncQdrantClient(url=settings.QDRANT_URL)

async def get_qdrant():
    """
    Dependency generator for FastAPI routes.
    Yields the Qdrant client for vector operations.
    """
    yield client