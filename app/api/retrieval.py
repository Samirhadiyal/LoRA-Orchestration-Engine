import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.retrieval.search import hybrid_search

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/retrieval", tags=["Retrieval"])


class SearchRequest(BaseModel):
    """Request body for the POST /search endpoint."""
    query: str = Field(..., min_length=1, description="The search prompt or question")
    limit: int = Field(default=5, ge=1, le=20, description="Max number of chunks to return")


class SearchResult(BaseModel):
    """A single search result chunk."""
    text: str
    doc_id: str = ""
    chunk_id: str = ""
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    """Structured response for search endpoints."""
    query: str
    count: int
    results: list[SearchResult]


def _execute_search(query: str, limit: int) -> SearchResponse:
    """Shared search logic with error handling for both GET and POST."""
    try:
        raw_results = hybrid_search(query=query, top_k=limit)
    except Exception as e:
        logger.error("Hybrid search failed (Qdrant may be unavailable): %s", e)
        raise HTTPException(
            status_code=503,
            detail=f"Knowledge base search unavailable: {e!s}"
        )

    results = [
        SearchResult(
            text=r.get("text", ""),
            doc_id=r.get("doc_id", ""),
            chunk_id=r.get("chunk_id", ""),
            score=r.get("score"),
            metadata=r.get("metadata", {}),
        )
        for r in raw_results
    ]

    return SearchResponse(query=query, count=len(results), results=results)


@router.get("/search")
async def search_knowledge_base_get(
    query: str = Query(..., description="The search prompt or question"),
    limit: int = Query(default=5, ge=1, le=20, description="Max number of chunks to return"),
) -> SearchResponse:
    """
    GET variant — Performs Hybrid Search (Dense + BM25 + Cross-Encoder Reranking).
    """
    return _execute_search(query, limit)


@router.post("/search")
async def search_knowledge_base_post(body: SearchRequest) -> SearchResponse:
    """
    POST variant — Performs Hybrid Search with a JSON request body.
    Preferred for complex or long queries.
    """
    return _execute_search(body.query, body.limit)