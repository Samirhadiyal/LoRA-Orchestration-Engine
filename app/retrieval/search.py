from typing import List, Dict, Any
from app.embeddings.bge import generate_embeddings
from app.retrieval.indexer import qdrant, COLLECTION_NAME

def semantic_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Converts a query string into a vector, searches Qdrant for matching 
    chunks, and returns formatted results with similarity scores.
    """
    if not query.strip():
        return []

    # 1. Convert the user's plain text query into a 384-dim vector
    query_vector = generate_embeddings([query])[0]

    # 2. Search Qdrant for nearest neighbors using the updated query_points API
    response = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k
    )

    # 3. Format hits into a clean list for the Planner / API
    formatted_results = []
    for hit in response.points:
        formatted_results.append({
            "score": round(float(hit.score), 4),
            "text": hit.payload.get("text", ""),
            "doc_id": hit.payload.get("doc_id", ""),
            "chunk_id": hit.payload.get("chunk_id", ""),
            "metadata": hit.payload.get("metadata", {})
        })

    return formatted_results