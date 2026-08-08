import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from app.embeddings.bge import generate_embeddings

# 1. Connect to the local Qdrant container running via Docker Compose
qdrant = QdrantClient(host="localhost", port=6333, check_compatibility=False)
COLLECTION_NAME = "neuromesh_knowledge"

def init_qdrant_collection():
    """
    Ensures the target collection exists in Qdrant with matching vector settings.
    """
    try:
        collections = qdrant.get_collections().collections
        exists = any(c.name == COLLECTION_NAME for c in collections)
        
        if not exists:
            qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
            print(f"Created Qdrant collection: '{COLLECTION_NAME}'")
    except Exception as e:
        print(f"Error initializing Qdrant collection: {e}")

def index_chunks(chunks: list[dict]) -> int:
    """
    Takes a list of text chunk dictionaries (passed from Engineer A's parser/chunker)
    generates vector embeddings, and upserts them into Qdrant.
    """
    if not chunks:
        return 0

    # Ensure collection is ready before pushing points
    init_qdrant_collection()

    # Extract all text strings to generate embeddings in a single batch
    texts = [chunk["text"] for chunk in chunks]
    vectors = generate_embeddings(texts)

    points = []
    for idx, chunk in enumerate(chunks):
        # Qdrant points require a unique ID (UUID string or integer)
        point_id = str(uuid.uuid4())
        
        points.append(
            PointStruct(
                id=point_id,
                vector=vectors[idx],
                payload={
                    "chunk_id": chunk.get("chunk_id", point_id),
                    "doc_id": chunk.get("doc_id", "unknown"),
                    "text": chunk["text"],
                    "metadata": chunk.get("metadata", {})
                }
            )
        )

    # Upsert points into Qdrant
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(points)