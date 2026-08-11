import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rank_bm25 import BM25Okapi

from app.embeddings.bge import generate_embeddings

qdrant = QdrantClient(host="localhost", port=6333, check_compatibility=False)
COLLECTION_NAME = "neuromesh_knowledge"

# In-memory store for BM25 sparse search
bm25_index = None
corpus_chunks = []

def init_qdrant_collection():
    try:
        collections = qdrant.get_collections().collections
        exists = any(c.name == COLLECTION_NAME for c in collections)
        if not exists:
            qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
    except Exception as e:  # noqa: BLE001
        print(f"Error initializing Qdrant collection: {e}")

def index_chunks(chunks: list[dict]) -> int:
    global bm25_index
    if not chunks:
        return 0

    init_qdrant_collection()

    # 1. Update Dense Index (Qdrant)
    texts = [chunk["text"] for chunk in chunks]
    vectors = generate_embeddings(texts)

    points = []
    for idx, chunk in enumerate(chunks):
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

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

    # 2. Update Sparse Index (BM25)
    corpus_chunks.extend(chunks)
    tokenized_corpus = [doc["text"].lower().split() for doc in corpus_chunks]
    bm25_index = BM25Okapi(tokenized_corpus)

    return len(points)
