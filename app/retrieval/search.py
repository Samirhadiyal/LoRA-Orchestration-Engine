import logging
from typing import Any

from sentence_transformers import CrossEncoder

from app.embeddings.bge import generate_embeddings
from app.retrieval.indexer import COLLECTION_NAME, bm25_index, corpus_chunks, qdrant

logger = logging.getLogger(__name__)

# Load Cross-Encoder model for reranking top results
try:
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
except (OSError, RuntimeError) as e:
    logger.warning("Failed to load Cross-Encoder model: %s", e)
    reranker = None

def reciprocal_rank_fusion(dense_results: list[dict], sparse_results: list[dict], k: int = 60) -> list[dict]:
    """
    Combines dense vector results and sparse BM25 results using Reciprocal Rank Fusion (RRF).
    """
    rrf_scores: dict[str, float] = {}
    docs: dict[str, dict] = {}

    for rank, item in enumerate(dense_results):
        doc_id = item["text"]
        docs[doc_id] = item
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    for rank, item in enumerate(sparse_results):
        doc_id = item["text"]
        if doc_id not in docs:
            docs[doc_id] = item
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    # Sort documents by accumulated RRF score
    reranked_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [docs[doc_id] for doc_id, _ in reranked_docs]

def hybrid_search(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """
    Executes Dense Search + BM25 Sparse Search -> RRF Fusion -> Cross-Encoder Reranking.
    """
    if not query.strip():
        return []

    # --- 1. Dense Vector Search ---
    query_vector = generate_embeddings([query])[0]
    qdrant_response = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=10
    )
    dense_hits: list[dict[str, Any]] = []
    for hit in qdrant_response.points:
        payload = hit.payload or {}
        dense_hits.append(
            {
                "text": payload.get("text", ""),
                "doc_id": payload.get("doc_id", ""),
                "chunk_id": payload.get("chunk_id", ""),
                "metadata": payload.get("metadata", {})
            }
        )

    # --- 2. Sparse BM25 Search ---
    sparse_hits: list[dict[str, Any]] = []
    if bm25_index and corpus_chunks:
        tokenized_query = query.lower().split()
        top_sparse_chunks = bm25_index.get_top_n(tokenized_query, corpus_chunks, n=10)
        sparse_hits = [
            {
                "text": chunk["text"],
                "doc_id": chunk.get("doc_id", ""),
                "chunk_id": chunk.get("chunk_id", ""),
                "metadata": chunk.get("metadata", {})
            }
            for chunk in top_sparse_chunks
        ]

    # --- 3. Reciprocal Rank Fusion (RRF) ---
    fused_candidates = reciprocal_rank_fusion(dense_hits, sparse_hits)

    if not fused_candidates:
        return []

    # --- 4. Cross-Encoder Reranking ---
    if reranker:
        pairs = [[query, candidate["text"]] for candidate in fused_candidates]
        scores = reranker.predict(pairs)

        for i, score in enumerate(scores):
            fused_candidates[i]["score"] = round(float(score), 4)

        # Sort candidates by Cross-Encoder relevance score
        final_results = sorted(fused_candidates, key=lambda x: x["score"], reverse=True)
    else:
        final_results = fused_candidates

    return final_results[:top_k]
